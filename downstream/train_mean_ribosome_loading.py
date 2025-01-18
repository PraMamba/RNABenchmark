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
from sklearn.metrics import roc_auc_score, matthews_corrcoef 

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from transformers import Trainer, TrainingArguments, AutoTokenizer, EsmTokenizer, AutoConfig, AutoModel, AutoModelForSequenceClassification, EarlyStoppingCallback

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
from model.dnabert2.bert_layers import DNABERT2ForSequenceClassification
from model.genalm.modeling_genalm import GENALMForSequenceClassification
from model.caduceus.modeling_caduceus import CaduceusForSequenceClassification
from model.nucleotide_transformer.modeling_nt import EsmForSequenceClassification
from model.evo.modeling_hyena import StripedHyenaForSequenceClassification
from model.hyenadna.modeling_hyena import HyenaDNAForSequenceClassification

from tokenizer.tokenization_opensource import OpenRnaLMTokenizer

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
	model_name_or_path: Optional[str] = field(
		default=None,
		metadata={
			"help": (
				"The path or identifier of the model checkpoint for initializing weights. "
				"If you want to train a model from scratch, leave this field empty. "
				"Typically used to load pre-trained models from a local path or Hugging Face Hub."
			)
		},
	)
	use_lora: bool = field(
		default=False,
		metadata={
			"help": (
				"Whether to use LoRA (Low-Rank Adaptation) for fine-tuning. "
				"LoRA can help reduce the number of trainable parameters, making the training more efficient."
			)
		},
	)
	use_alibi: bool = field(
		default=True,
		metadata={
			"help": (
				"Whether to use ALiBi (Attention with Linear Biases) mechanism. "
				"This is typically used to improve the model's performance on long sequence inputs."
			)
		},
	)
	use_features: bool = field(
		default=True,
		metadata={
			"help": (
				"Whether to use additional sequence features, such as structural or chemical properties, "
				"as input to the model."
			)
		},
	)
	lora_r: int = field(
		default=8,
		metadata={
			"help": (
				"The rank (r) for the low-rank decomposition in LoRA. "
				"Higher values may capture more complex representations but increase the parameter count."
			)
		},
	)
	lora_alpha: int = field(
		default=32,
		metadata={
			"help": (
				"Scaling factor (alpha) for LoRA. "
				"This affects the scaling of the adapted weights during training."
			)
		},
	)
	lora_dropout: float = field(
		default=0.05,
		metadata={
			"help": (
				"Dropout rate for LoRA layers to prevent overfitting. "
				"Recommended values are in the range [0.0, 0.2]."
			)
		},
	)
	lora_target_modules: str = field(
		default="query,value",
		metadata={
			"help": (
				"Specifies which modules (e.g., attention query/value layers) should be adapted using LoRA. "
				"Provide a comma-separated list of module names."
			)
		},
	)
	tokenizer_name_or_path: Optional[str] = field(
		default="",
		metadata={
			"help": (
				"The path or identifier of the tokenizer to use. "
				"If not provided, the tokenizer will default to the model's tokenizer (if available)."
			)
		},
	)
	model_max_length: int = field(
		default=512,
		metadata={
			"help": (
				"The maximum total input sequence length after tokenization. "
				"Sequences longer than this will be truncated. "
				"This parameter is critical for memory management during training."
			)
		},
	)
	eval_and_save_results: bool = field(
		default=True,
		metadata={
			"help": (
				"Whether to evaluate the model and save results during or after training. "
				"If disabled, evaluation metrics will not be calculated."
			)
		},
	)
	model_type: str = field(
		default="rna",
		metadata={
			"help": (
				"Specifies the type of model architecture to use. "
				"For example: 'rna' for RNA-related tasks or other model-specific types."
			)
		},
	)
	token_type: str = field(
		default="6mer",
		metadata={
			"help": (
				"The type of tokenization used for input sequences. "
				"For example: '6mer' represents k-mer tokens of length 6."
			)
		},
	)
	train_from_scratch: bool = field(
		default=False,
		metadata={
			"help": (
				"Whether to train the model from scratch. "
				"If set to True, model weights will not be initialized from a pre-trained checkpoint."
			)
		},
	)
	attn_implementation: Optional[str] = field(
		default="eager",
		metadata={
			"help": (
				"The attention mechanism implementation to use in the model. "
				"Options include 'eager' for standard attention, 'sdpa' for scaled dot-product attention, "
				"and 'flash_attention_2' for optimized memory-efficient attention."
			),
			"choices": ["eager", "sdpa", "flash_attention_2"],
		},
	)
	trust_remote_code: bool = field(
		default=False,
		metadata={
			"help": (
				"Whether to trust and execute remote code from the Hugging Face Hub. "
				"Only enable this option for trusted repositories after reviewing their code."
			)
		},
	)
	cache_dir: Optional[str] = field(
		default=None,
		metadata={
			"help": (
				"The directory where pre-trained models and tokenizer files from Hugging Face Hub will be stored locally."
			)
		},
	)


@dataclass
class DataArguments:
	data_path: str = field(
		default=None,
		metadata={
			"help": (
				"Path to the directory containing the dataset files. "
				"This path should include training, validation, and test datasets, depending on the task."
			)
		},
	)
	kmer: int = field(
		default=-1,
		metadata={
			"help": (
				"The k-mer length for input sequence tokenization. "
				"Set to -1 to disable k-mer tokenization, or specify a positive integer for the k-mer size."
			)
		},
	)
	data_train_path: str = field(
		default=None,
		metadata={
			"help": (
				"Path to the training data file. "
				"The file should contain sequences and corresponding labels formatted for the task."
			)
		},
	)
	data_val_path: str = field(
		default=None,
		metadata={
			"help": (
				"Path to the validation data file. "
				"This file is used to evaluate the model's performance during training."
			)
		},
	)
	data_test_path: str = field(
		default=None,
		metadata={
			"help": (
				"Path to the test data file(s). "
				"This can be a list of paths for multiple test datasets. "
				"The test dataset is used to evaluate the final model."
			)
		},
	)


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

	def __init__(self, data_path: str, args, tokenizer: transformers.PreTrainedTokenizer, kmer: int=-1):

		super(SupervisedDataset, self).__init__()
		
		# load data from the disk
		with open(data_path, "r") as f:
			data = list(csv.reader(f))[1:]

		if len(data[0]) == 2:
			texts = [d[0].upper().replace("U", "T") for d in data]          
			labels = [float(d[1]) for d in data]
			
		else:
			print(len(data[0]))
			raise ValueError("Data format not supported.")
		
		labels = np.array(labels)
		labels = labels.tolist()
		
		if kmer != -1:
			# only write file on the first process
			if torch.distributed.get_rank() not in [0, -1]:
				torch.distributed.barrier()

			logging.warning(f"Using {kmer}-mer as input.")
			texts = load_or_generate_kmer(data_path, texts, kmer)
			
			if torch.distributed.get_rank() == 0:
				torch.distributed.barrier()
				
		# ensure tokenier
		logger.info(texts[0])
		test_example = tokenizer.tokenize(texts[0])
		logger.info(test_example)
		logger.info(len(test_example))
		logger.info(tokenizer(texts[0]))
		
		output = tokenizer\
			(
				texts, 
				return_tensors="pt",
				padding="max_length",
				max_length=tokenizer.model_max_length,
				truncation=True,
				return_attention_mask=True,
			)

		self.input_ids = output["input_ids"]
		self.attention_mask = output["attention_mask"]
		self.labels = labels
		self.num_labels = 1

	def __len__(self):
		return len(self.input_ids)

	def __getitem__(self, i) -> Dict[str, torch.Tensor,]:
		return dict(input_ids=self.input_ids[i], labels=self.labels[i], attention_mask=self.attention_mask[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
	"""Collate examples for supervised fine-tuning."""

	def __init__(self, tokenizer: transformers.PreTrainedTokenizer, args):
		self.tokenizer = tokenizer
		self.args = args

	def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
		
		input_ids, labels, attention_mask = tuple([instance[key] for instance in instances] for key in ("input_ids" ,"labels", "attention_mask"))
		attention_mask = torch.stack(attention_mask)
		input_ids = torch.stack(input_ids)
		labels = torch.Tensor(labels).float()  

		return dict\
      	(
			input_ids=input_ids,
			labels=labels,
			attention_mask=attention_mask,
		)


"""
Manually calculate the mse and r^2.
"""
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    
	# 如果 logits 是 tuple，可能需要提取实际 logits 部分
	if isinstance(logits, tuple):
		logits = logits[0]  # 假设第一个元素是 logits
    
	labels = labels.squeeze()
	logits = logits.squeeze()
 
	return {"MSE": sklearn.metrics.mean_squared_error(labels, logits), "R^2" : scipy.stats.pearsonr(labels, logits)[0]**2,}


"""
Compute metrics used for huggingface trainer.
"""
def compute_metrics(eval_pred):
	logits, labels = eval_pred
	return calculate_metric_with_sklearn(logits, labels)


SUPPORT_CUSTOMED_MODEL_TYPE = \
	[
		'RNA-FM',
		'RNABERT',
		'RNA-MSM',
		'SpliceBERT-Human510','SpliceBERT-MS510','SpliceBERT-MS1024',
		'UTRBERT-3mer','UTRBERT-4mer','UTRBERT-5mer','UTRBERT-6mer',
		'UTR-LM-MRL','UTR-LM-TE-EL', 
		'DNABERT-2', 
		'GENA-LM',
		
	]
	
SUPPORT_AUTO_MODEL_TYPE = \
	[
		'Caduceus',
		'Nucleotide-Transformer',
		# 'Evo',
		'Mistral-DNA',
		# 'HyenaDNA',
		
	]


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
	if model_args.model_type == 'RNALM':
		tokenizer = EsmTokenizer.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			model_max_length=model_args.model_max_length,
			padding_side="right",
			use_fast=True,
			trust_remote_code=model_args.trust_remote_code,
		)
	elif model_args.model_type in SUPPORT_CUSTOMED_MODEL_TYPE:
		tokenizer = OpenRnaLMTokenizer.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			model_max_length=model_args.model_max_length,
			padding_side="right",
			use_fast=True,
			trust_remote_code=model_args.trust_remote_code,
		)
	elif model_args.model_type in SUPPORT_AUTO_MODEL_TYPE:
		tokenizer = AutoTokenizer.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			model_max_length=model_args.model_max_length,
			trust_remote_code=model_args.trust_remote_code,
		)
	elif model_args.model_type == 'Evo':
		tokenizer = AutoTokenizer.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			model_max_length=model_args.model_max_length,
			trust_remote_code=model_args.trust_remote_code,
			cls_token="@",
			eos_token="&",
			bos_token="^",
			pad_token='N',
		)
	elif model_args.model_type == 'HyenaDNA':
		tokenizer = AutoTokenizer.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			model_max_length=model_args.model_max_length,
			trust_remote_code=model_args.trust_remote_code,
			padding_side="right",
		)
	else:
		tokenizer = AutoTokenizer.from_pretrained(
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
		data_args.kmer = int(model_args.token_type[0])
  
  
	# load datasets and data collator
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
	if model_args.model_type == 'RNALM':
		if model_args.train_from_scratch:
			logger.info('Train from scratch')
			config = RnaLmConfig.from_pretrained(model_args.model_name_or_path,
				num_labels=train_dataset.num_labels,
				token_type=model_args.token_type,
				problem_type="regression",
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
			problem_type="regression",
			trust_remote_code=model_args.trust_remote_code,
		)
	elif model_args.model_type == 'RNABERT':
		logger.info(f'Loading {model_args.model_type}')
		model = RnaBertForSequenceClassification.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="regression",
			trust_remote_code=model_args.trust_remote_code,
		)
	elif model_args.model_type == 'RNA-MSM':
		logger.info(f'Loading {model_args.model_type}')
		model = RnaMsmForSequenceClassification.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="regression",
			trust_remote_code=model_args.trust_remote_code,
		)
	elif model_args.model_type == 'SpliceBERT':
		logger.info(f'Loading {model_args.model_type}')
		model = SpliceBertForSequenceClassification.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="regression",
			trust_remote_code=model_args.trust_remote_code,
		)
	elif model_args.model_type == 'UTRBERT':
		logger.info(f'Loading {model_args.model_type}')
		model = UtrBertForSequenceClassification.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="regression",
			trust_remote_code=model_args.trust_remote_code,
		)
	elif model_args.model_type == 'UTR-LM':
		logger.info(f'Loading {model_args.model_type}')
		model = UtrLmForSequenceClassification.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="regression",
			trust_remote_code=model_args.trust_remote_code,
		)
	elif model_args.model_type == 'DNABERT-2':
		logger.info(f'Loading {model_args.model_type}')
		model = DNABERT2ForSequenceClassification.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="regression",
			trust_remote_code=model_args.trust_remote_code,
		)
	elif model_args.model_type == 'GENA-LM':
		logger.info(f'Loading {model_args.model_type}')
		model = GENALMForSequenceClassification.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="regression",
			trust_remote_code=model_args.trust_remote_code,
		)
	elif model_args.model_type == 'Caduceus':
		logger.info(f'Loading {model_args.model_type}')
		model = CaduceusForSequenceClassification.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="regression",
			trust_remote_code=model_args.trust_remote_code,
			ignore_mismatched_sizes=True,
		)
		training_args.save_safetensors = False  # Attention: The weights trying to be saved contained shared tensors, save model weight in *.bin
	elif model_args.model_type == 'Nucleotide-Transformer':
		logger.info(f'Loading {model_args.model_type}')
		model = EsmForSequenceClassification.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="regression",
			trust_remote_code=model_args.trust_remote_code,
		)
	elif model_args.model_type == 'Evo':
		logger.info(f'Loading {model_args.model_type}')
		model = StripedHyenaForSequenceClassification.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="regression",
			trust_remote_code=model_args.trust_remote_code,
		)
		model.config.use_cache = False
	elif model_args.model_type == 'Mistral-DNA':
		logger.info(f'Loading {model_args.model_type}')
		model = AutoModelForSequenceClassification.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="regression",
			trust_remote_code=model_args.trust_remote_code,
		)
		if tokenizer.pad_token is None:
			tokenizer.pad_token = tokenizer.eos_token
		model.config.pad_token_id = tokenizer.pad_token_id
	elif model_args.model_type == 'HyenaDNA':
		logger.info(f'Loading {model_args.model_type}')
		model = HyenaDNAForSequenceClassification.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="regression",
			trust_remote_code=model_args.trust_remote_code,
		)
		training_args.save_safetensors = False  # Attention: The weights trying to be saved contained shared tensors, save model weight in *.bin

	early_stopping = EarlyStoppingCallback(early_stopping_patience=5)
	
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
		logger.info(f"Evaluation Result On The Test Set: {results}")

		with open(os.path.join(results_path, "test_results.json"), "w") as f:
			json.dump(results, f, indent=4)


if __name__ == "__main__":
	main()
