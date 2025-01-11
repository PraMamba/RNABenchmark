import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List

import torch
import random
import sklearn
import scipy
import transformers

import numpy as np
from torch.utils.data import Dataset
import pdb

from transformers import Trainer, TrainingArguments, BertTokenizer, EsmTokenizer, EsmModel, AutoConfig, AutoModel, EarlyStoppingCallback

import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)

from model.rnalm.modeling_rnalm import RnaLmForSequenceClassification
from model.rnalm.rnalm_config import RnaLmConfig
from model.rnafm.modeling_rnafm import RnaFmForSequenceClassification
from model.rnabert.modeling_rnabert import RnaBertForSequenceClassification
from model.rnamsm.modeling_rnamsm import RnaMsmForSequenceClassification
from model.splicebert.modeling_splicebert import SpliceBertForSequenceClassification
from model.utrbert.modeling_utrbert import UtrBertForSequenceClassification
from model.utrlm.modeling_utrlm import UtrLmForSequenceClassification
from tokenizer.tokenization_opensource import OpenRnaLMTokenizer

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
	model_name_or_path: Optional[str] = field(
		default=None,
		metadata={
			"help": (
				"The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
			)
		},
	)
	use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
	use_alibi: bool = field(default=True, metadata={"help": "whether to use alibi"})
	use_features: bool = field(default=True, metadata={"help": "whether to use alibi"})
	lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
	lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
	lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
	lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})
	tokenizer_name_or_path: Optional[str] = field(default="")
	model_max_length: int = field(default=512, metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."})
	checkpointing: bool = field(default=False)
	eval_and_save_results: bool = field(default=True)
	stage: str = field(default='0')
	model_type: str = field(default='rna')
	token_type: str = field(default='6mer')
	train_from_scratch: bool = field(default=False)
	attn_implementation: Optional[str] = field(
		default="eager",
		metadata={
			"help": (
				"The attention implementation to use in the model (if relevant)."
			),
			"choices": ["eager", "sdpa", "flash_attention_2"],
		},
	)
	trust_remote_code: bool = field(
		default=False,
		metadata={
			"help": (
				"Whether to trust the execution of code from datasets/models defined on the Hub."
				" This option should only be set to `True` for repositories you trust and in which you have read the"
				" code, as it will execute code present on the Hub on your local machine."
			)
		},
	)
	cache_dir: Optional[str] = field(
		default=None,
		metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
	)


@dataclass
class DataArguments:
	data_path: str = field(default=None, metadata={"help": "Path to the training data."})
	kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})
	data_train_path: str = field(default=None, metadata={"help": "Path to the training data."})
	data_val_path: str = field(default=None, metadata={"help": "Path to the training data."})
	data_test_path: str = field(default=None, metadata={"help": "Path to the test data. is list"})


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
	"""Collects the state dict and dump to disk."""
	state_dict = trainer.model.state_dict()
	if trainer.args.should_save:
		cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
		del state_dict
		trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.distributed.get_rank() >= 0:
		torch.cuda.manual_seed_all(args.seed)


"""
Transform a rna sequence to k-mer string
"""
def generate_kmer_str(sequence: str, k: int) -> str:
	"""Generate k-mer string from rna sequence."""
	return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


"""
Load or generate k-mer string for each rna sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""
def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
	"""Load or generate k-mer string for each rna sequence."""
	kmer_path = data_path.replace(".csv", f"_{k}mer.json")
	if os.path.exists(kmer_path):
		logger.info(f"Loading k-mer from {kmer_path}...")
		with open(kmer_path, "r") as f:
			kmer = json.load(f)
	else:
		logger.info(f"Generating k-mer...")
		kmer = [generate_kmer_str(text, k) for text in texts]
		with open(kmer_path, "w") as f:
			logger.info(f"Saving k-mer to {kmer_path}...")
			json.dump(kmer, f)

	return kmer

class SupervisedDataset(Dataset):
	"""Dataset for supervised fine-tuning."""

	def __init__(self, data_path: str, args, tokenizer: transformers.PreTrainedTokenizer, kmer: int = -1):
		super(SupervisedDataset, self).__init__()

		# load data from the disk
		with open(data_path, "r") as f:
			data = list(csv.reader(f))[1:]
		if len(data[0]) == 2:
			# data is in the format of [text, label]
			logger.info("Perform single sequence classification.")
			texts = [d[0].upper().replace("U", "T") for d in data]
			labels = [int(d[1]) for d in data]
		else:
			logger.info(len(data[0]))
			raise ValueError("Data format not supported.")
		text = texts[0]

		if kmer != -1:
			# only write file on the first process
			if torch.distributed.get_rank() not in [0, -1]:
				torch.distributed.barrier()

			logger.info(f"Using {kmer}-mer as input.")
			texts = load_or_generate_kmer(data_path, texts, kmer)

			if torch.distributed.get_rank() == 0:
				torch.distributed.barrier()
    
		# ensure tokenier
		logger.info(texts[0])
		test_example = tokenizer.tokenize(texts[0])
		logger.info(test_example)
		logger.info(len(test_example))
		logger.info(tokenizer(texts[0]))
		self.labels = labels
		self.num_labels = len(set(labels))
		self.texts = texts

	def __len__(self):
		return len(self.texts)

	def __getitem__(self, i) -> Dict[str, torch.Tensor]:
		return dict(input_ids=self.texts[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
	def __init__(self, tokenizer: transformers.PreTrainedTokenizer, args):
		self.tokenizer = tokenizer
		self.args = args

	def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

		seqs, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

		output = self.tokenizer(seqs, padding='longest', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
		input_ids = output["input_ids"]
		attention_mask = output["attention_mask"]
		labels = torch.Tensor(labels).long()
		return dict(
			input_ids=input_ids,
			labels=labels,
			attention_mask=attention_mask,
		)

"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
	predictions = np.argmax(logits, axis=-1)
	return {
		"accuracy": sklearn.metrics.accuracy_score(labels, predictions),
		"f1": sklearn.metrics.f1_score(labels, predictions, average="macro", zero_division=0),
		"matthews_correlation": sklearn.metrics.matthews_corrcoef(labels, predictions),
		"precision": sklearn.metrics.precision_score(labels, predictions, average="macro", zero_division=0),
		"recall": sklearn.metrics.recall_score(labels, predictions, average="macro", zero_division=0),
	}

"""
Compute metrics used for huggingface trainer.
"""
def compute_metrics(eval_pred):
	logits, labels = eval_pred
	return calculate_metric_with_sklearn(logits, labels)

def get_parameter_number(model):
	total_num = sum(p.numel() for p in model.parameters())
	trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
	return {'Total': total_num, 'Trainable': trainable_num}

def main():
	parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
	model_args, data_args, training_args = parser.parse_args_into_dataclasses()
	set_seed(training_args)
 
	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		handlers=[logging.StreamHandler(sys.stdout)],
	)

	if training_args.should_log:
		# The default of training_args.log_level is passive, so we set log level at info here to have that default.
		transformers.utils.logging.set_verbosity_info()

	log_level = training_args.get_process_log_level()
	logger.setLevel(log_level)
	transformers.utils.logging.set_verbosity(log_level)
	transformers.utils.logging.enable_default_handler()
	transformers.utils.logging.enable_explicit_format()
 
	# load tokenizer
	if model_args.model_type == 'rnalm':
		tokenizer = EsmTokenizer.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			model_max_length=model_args.model_max_length,
			padding_side="right",
			use_fast=True,
			trust_remote_code=model_args.trust_remote_code,
		)
	elif model_args.model_type in ['RNA-FM','RNABERT','RNA-MSM','SpliceBERT-Human510','SpliceBERT-MS510','SpliceBERT-MS1024','UTRBERT-3mer','UTRBERT-4mer','UTRBERT-5mer','UTRBERT-6mer','UTR-LM-MRL','UTR-LM-TE-EL']:
		tokenizer = OpenRnaLMTokenizer.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			model_max_length=model_args.model_max_length,
			padding_side="right",
			use_fast=True,
			trust_remote_code=model_args.trust_remote_code,
		)
	else:
		tokenizer = transformers.AutoTokenizer.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			model_max_length=model_args.model_max_length,
			padding_side="right",
			use_fast=True,
			trust_remote_code=model_args.trust_remote_code,
		)


	if "InstaDeepAI" in model_args.model_name_or_path:
		tokenizer.eos_token = tokenizer.pad_token
	if 'mer' in model_args.token_type:
		data_args.kmer=int(model_args.token_type[0])
  
	# define datasets and data collator
	train_dataset = SupervisedDataset\
	(
		tokenizer=tokenizer, 
		args=training_args,
		data_path=os.path.join(data_args.data_path, data_args.data_train_path),
		kmer=data_args.kmer
  	)
 
	val_dataset = SupervisedDataset\
	(
		tokenizer=tokenizer, 
		args=training_args,
		data_path=os.path.join(data_args.data_path, data_args.data_val_path),
		kmer=data_args.kmer
	)

	test_dataset = SupervisedDataset\
	(
		tokenizer=tokenizer, 
		args=training_args,
		data_path=os.path.join(data_args.data_path, data_args.data_test_path),
		kmer=data_args.kmer
	)
 
	data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer,args=training_args)
	logger.info(f'train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}')

	# load model
	if model_args.model_type == 'rnalm':
		if model_args.train_from_scratch:
			logger.info('Train from scratch')
			config = RnaLmConfig.from_pretrained(model_args.model_name_or_path,
				num_labels=train_dataset.num_labels,
				token_type=model_args.token_type,
				problem_type="single_label_classification",
				attn_implementation=model_args.attn_implementation,
				)
			logger.info(config)
			model = RnaLmForSequenceClassification(config)
		else:
			logger.info('Loading RNALM')
			logger.info(f'train_dataset num_labels: {train_dataset.num_labels}')
			model = RnaLmForSequenceClassification.from_pretrained(
				model_args.model_name_or_path,
				cache_dir=model_args.cache_dir,
				num_labels=train_dataset.num_labels,
				trust_remote_code=model_args.trust_remote_code,
				token_type=model_args.token_type,
				attn_implementation=model_args.attn_implementation,
				)
	elif model_args.model_type == 'RNA-FM':
		logger.info(f'Loading {model_args.model_type}')
		model = RnaFmForSequenceClassification.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="single_label_classification",
			trust_remote_code=model_args.trust_remote_code,
		)
	elif model_args.model_type == 'RNABERT':
		logger.info(f'Loading {model_args.model_type}')
		model = RnaBertForSequenceClassification.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="single_label_classification",
			trust_remote_code=model_args.trust_remote_code,
		)
	elif model_args.model_type == 'RNA-MSM':
		logger.info(f'Loading {model_args.model_type}')
		model = RnaMsmForSequenceClassification.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="single_label_classification",
			trust_remote_code=model_args.trust_remote_code,
		)
	elif 'SpliceBERT' in model_args.model_type:
		logger.info(f'Loading {model_args.model_type}')
		model = SpliceBertForSequenceClassification.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="single_label_classification",
			trust_remote_code=model_args.trust_remote_code,
		)
	elif 'UTRBERT' in model_args.model_type:
		logger.info(f'Loading {model_args.model_type}')
		model = UtrBertForSequenceClassification.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="single_label_classification",
			trust_remote_code=model_args.trust_remote_code,
		)
	elif 'UTR-LM' in model_args.model_type:
		logger.info(f'Loading {model_args.model_type}')
		model = UtrLmForSequenceClassification.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="single_label_classification",
			trust_remote_code=model_args.trust_remote_code,
		)


	early_stopping = EarlyStoppingCallback(early_stopping_patience=20)
	trainer = Trainer\
		(
			model=model,
			processing_class=tokenizer,
			args=training_args,
			compute_metrics=compute_metrics,
			train_dataset=train_dataset,
			eval_dataset=val_dataset,
			data_collator=data_collator,
			callbacks=[early_stopping],
		)
	trainer.train()


	if model_args.eval_and_save_results:
		trainer.save_state()
		safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
     
		results_path = training_args.output_dir
		results = trainer.evaluate(eval_dataset=test_dataset)
		logger.info("Evaluation Result On The Test Set:", results)

		with open(os.path.join(results_path, "test_results.json"), "w") as f:
			json.dump(results, f, indent=4)


if __name__ == "__main__":
	main()

