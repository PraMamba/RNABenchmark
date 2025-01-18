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
from tqdm import tqdm
from torch.utils.data import Dataset

from transformers import Trainer, TrainingArguments, AutoTokenizer, EsmTokenizer, AutoConfig, AutoModel, AutoModelForSequenceClassification, EarlyStoppingCallback

import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)

from model.rnalm.modeling_rnalm import RnaLmForNucleotideLevel
from model.rnalm.rnalm_config import RnaLmConfig
from model.rnafm.modeling_rnafm import RnaFmForNucleotideLevel
from model.rnabert.modeling_rnabert import RnaBertForNucleotideLevel
from model.rnamsm.modeling_rnamsm import RnaMsmForNucleotideLevel
from model.splicebert.modeling_splicebert import SpliceBertForNucleotideLevel
from model.utrbert.modeling_utrbert import UtrBertForNucleotideLevel
from model.utrlm.modeling_utrlm import UtrLmForNucleotideLevel

from model.dnabert2.bert_layers import DNABERT2ForNucleotideLevel
from model.genalm.modeling_genalm import GENALMForNucleotideLevel
from model.caduceus.modeling_caduceus import CaduceusForNucleotideLevel
from model.nucleotide_transformer.modeling_nt import EsmForNucleotideLevel
from model.evo.modeling_hyena import StripedHyenaForNucleotideLevel
from model.hyenadna.modeling_hyena import HyenaDNAForNucleotideLevel

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


def bpe_position(texts,attn_mask, tokenizer):
	position_id = torch.zeros(attn_mask.shape)
	for i,text in enumerate(texts):   
		text = tokenizer.tokenize(text)
		position_id[:, 0] = 1 #[cls]
		index = 0
		for j, token in enumerate(text):
			index = j+1
			position_id[i,index] = len(token) # start after [cls]   
		position_id[i, index+1] = 1 # [sep]
		
	print(position_id[0,:])
	print('position_id.shape',position_id.shape)
	
	return position_id


class SupervisedDataset(Dataset):
	def __init__(self, data_path, tokenizer,signal_noise_cutoff, test_set=None, kmer=-1,args=None):
		super().__init__()
		self.df = pd.read_json(data_path)
		print('pre',self.df.shape)
		deg_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']
		
		self.is_test = test_set is not None or deg_cols[0] not in self.df.columns
		if self.is_test:
			self.df = self.df.query(("seq_length == 107" if test_set == 'public' else "seq_length == 130"))
			self.y = None
		else:
			self.df = self.df[self.df.signal_to_noise >= signal_noise_cutoff]
			self.y = np.stack([np.stack(self.df[col].values) for col in deg_cols], axis=-1)
			
		print('post', self.df.shape)
		self.sample_ids = self.df['id'].values
		texts = [d.upper().replace("U", "T") for d in self.df['sequence']]
			   
		seq_length = len(texts[0])
		if kmer != -1:
			# only write file on the first process
			if torch.distributed.get_rank() not in [0, -1]:
				torch.distributed.barrier()

			logging.warning(f"Using {kmer}-mer as input...")
			texts = load_or_generate_kmer(data_path, texts, kmer, test_set)

			if torch.distributed.get_rank() == 0:
				torch.distributed.barrier()
				
		# ensure tokenizer
		logger.info(texts[0])
		test_example = tokenizer.tokenize(texts[0])
		logger.info(test_example)
		logger.info(len(test_example))
		logger.info(tokenizer(texts[0]))
		
		output = self.tokenizer\
			(
				texts, 
				return_tensors="pt",
				padding="max_length",
				max_length=self.tokenizer.model_max_length,
				truncation=True,
				return_attention_mask=True,
			)
   
		self.input_ids = output["input_ids"]
		self.texts = texts
		# make sure the length of sequences in the dataset is the same
		self.weight_mask = torch.ones((self.input_ids.shape[0],seq_length+2))

		self.attention_mask = output["attention_mask"]
		if 'mer' in args.token_type:
			for i in range(1,kmer-1):
				self.weight_mask[:, i+1] = self.weight_mask[:, -i-2] = 1/(i+1) 
			self.weight_mask[:, kmer:-kmer] = 1/kmer
		self.post_token_length = torch.zeros(self.attention_mask.shape)
  
		if args.token_type == 'bpe' or args.token_type == 'non-overlap':
			self.post_token_length = bpe_position(self.texts,self.attention_mask,tokenizer)
   
		self.num_labels = 3
  
	def __getitem__(self, index: int):
		if self.is_test:          
			sample_id = self.sample_ids[index]
			return dict(input_ids=self.input_ids[index], sample_ids=sample_id, attention_mask=self.attention_mask[index],
				weight_mask=self.weight_mask[index],post_token_length=self.post_token_length[index])
		targets = torch.tensor(self.y[index, :, :], dtype=torch.float32)
  
		return dict\
		(
			input_ids=self.input_ids[index], 
			labels=targets, 
			attention_mask=self.attention_mask[index],
			weight_mask=self.weight_mask[index],
			post_token_length=self.post_token_length[index]
		)
	 
	
	def __len__(self) -> int:
		return self.df.shape[0]


@dataclass
class TestDataCollatorForSupervisedDataset(object):
	"""Collate examples for supervised fine-tuning."""

	tokenizer: transformers.PreTrainedTokenizer

	def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
		input_ids, sample_ids, attention_mask, weight_mask, post_token_length = tuple([instance[key] for instance in instances] for key in ("input_ids" ,"sample_ids", "attention_mask","weight_mask","post_token_length"))
		input_ids = torch.stack(input_ids)
		attention_mask = torch.stack(attention_mask)
		weight_mask = torch.stack(weight_mask)
		post_token_length = torch.stack(post_token_length)
		
		return dict(
			input_ids=input_ids,
			sample_ids=sample_ids,
			attention_mask=attention_mask,
			weight_mask=weight_mask,
			post_token_length=post_token_length
		)
  

@dataclass
class DataCollatorForSupervisedDataset(object):
	"""Collate examples for supervised fine-tuning."""

	tokenizer: transformers.PreTrainedTokenizer

	def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
		input_ids, labels, attention_mask, weight_mask, post_token_length  = tuple([instance[key] for instance in instances] for key in ("input_ids" ,"labels", "attention_mask","weight_mask","post_token_length"))
		input_ids = torch.stack(input_ids)
		labels = torch.stack(labels)
		attention_mask = torch.stack(attention_mask)
		weight_mask = torch.stack(weight_mask)
		post_token_length = torch.stack(post_token_length)
		return dict(
			input_ids=input_ids,
			labels=labels,
			attention_mask=attention_mask,
			weight_mask=weight_mask,
			post_token_length=post_token_length
		)


def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
	def rmse(labels,logits):
		return np.mean(np.square(labels - logits + 1e-6))
	score = 0
	num_scored = 3
	for i in range(num_scored):
		score += rmse(labels[:, :, i], logits[:, :, i]) / num_scored       
	return {"MCRMSE": score}


def compute_metrics(eval_pred):
	logits, labels = eval_pred
	return calculate_metric_with_sklearn(logits, labels)


def build_submission_df(ids, pred_tensor):
	if type(pred_tensor).__module__ != np.__name__:
		pred_tensor = pred_tensor.cpu().detach().numpy()
	res = []
	for i, id in enumerate(ids):
		
		for j, pred in enumerate(pred_tensor[i, :, :]):
			res.append([id+'_'+str(j)] + list(pred))
	return res


def make_pred_file(args, model, loaders, postfix=''):
	res = []
	model.to(args.device)
	print(args.device)
	model.eval()
	for eval_dataloader in loaders:
		for batch in tqdm(eval_dataloader, desc="Evaluating"):
			input_ids = batch["input_ids"].to(args.device)
			attention_mask = batch["attention_mask"].to(args.device)
			sample_ids = batch["sample_ids"]
			weight_mask = batch["weight_mask"].to(args.device)
			post_token_length = batch["post_token_length"].to(args.device)
			with torch.no_grad():
				test_pred = model(input_ids=input_ids, attention_mask=attention_mask,weight_mask=weight_mask, post_token_length=post_token_length)
				test_pred = test_pred[0][:, 1:-1,:] #exclude [cls] and [sep]
				res += build_submission_df(sample_ids, test_pred)

	pred_df = pd.DataFrame(res, columns=['id_seqpos', 'reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'])
	pred_df['deg_pH10'] = 0
	pred_df['deg_50C'] = 0
	results_path = os.path.join(args.output_dir, "results", args.run_name)
 
	print(results_path)
	os.makedirs(results_path, exist_ok=True)
 
	results_path = os.path.join(results_path, 'submission_'+postfix+'.csv')
	pred_df.to_csv(results_path, index=False)


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
  

	train_dataset = SupervisedDataset\
		(
			os.path.join(data_args.data_path, data_args.data_train_path), 
			tokenizer, 
			signal_noise_cutoff=0.6, 
			test_set=None, 
			kmer=data_args.kmer, 
			args=training_args
      	)
  
	val_dataset = SupervisedDataset\
		(
			os.path.join(data_args.data_path, data_args.data_val_path), 
			tokenizer, 
			signal_noise_cutoff=1.0, 
			test_set=None, 
			kmer=data_args.kmer, 
			args=training_args
    	)

	public_test_dataset = SupervisedDataset\
		(
			os.path.join(data_args.data_path, data_args.data_test_path), 
			tokenizer, 
			signal_noise_cutoff=-99.0, 
			test_set='public', 
			kmer=data_args.kmer, 
			args=training_args
		)

	private_test_dataset = SupervisedDataset\
		(
			os.path.join(data_args.data_path, data_args.data_test_path), 
			tokenizer, 
			signal_noise_cutoff=-99.0, 
			test_set='private', 
			kmer=data_args.kmer, 
			args=training_args
		)

	data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
	test_data_collator = TestDataCollatorForSupervisedDataset(tokenizer=tokenizer)
 
	logger.info(f'train: {len(train_dataset)}, val:{len(val_dataset)}, test:{len(private_test_dataset)}+{len(private_test_dataset)}')

	
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
			model = RnaLmForNucleotideLevel(config)
		else:
			logger.info('Loading RNALM')
			logger.info(f'train_dataset num_labels: {train_dataset.num_labels}')
			model = RnaLmForNucleotideLevel.from_pretrained(
				model_args.model_name_or_path,
				cache_dir=model_args.cache_dir,
				num_labels=train_dataset.num_labels,
				trust_remote_code=model_args.trust_remote_code,
				token_type=model_args.token_type,
				attn_implementation=model_args.attn_implementation,
				)
	elif model_args.model_type == 'RNA-FM':
		logger.info(f'Loading {model_args.model_type}')
		model = RnaFmForNucleotideLevel.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="regression",
			trust_remote_code=model_args.trust_remote_code,
		)
	elif model_args.model_type == 'RNABERT':
		logger.info(f'Loading {model_args.model_type}')
		model = RnaBertForNucleotideLevel.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="regression",
			trust_remote_code=model_args.trust_remote_code,
		)
	elif model_args.model_type == 'RNA-MSM':
		logger.info(f'Loading {model_args.model_type}')
		model = RnaMsmForNucleotideLevel.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="regression",
			trust_remote_code=model_args.trust_remote_code,
		)
	elif model_args.model_type == 'SpliceBERT':
		logger.info(f'Loading {model_args.model_type}')
		model = SpliceBertForNucleotideLevel.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="regression",
			trust_remote_code=model_args.trust_remote_code,
		)
	elif model_args.model_type == 'UTRBERT':
		logger.info(f'Loading {model_args.model_type}')
		model = UtrBertForNucleotideLevel.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="regression",
			trust_remote_code=model_args.trust_remote_code,
		)
	elif model_args.model_type == 'UTR-LM':
		logger.info(f'Loading {model_args.model_type}')
		model = UtrLmForNucleotideLevel.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="regression",
			trust_remote_code=model_args.trust_remote_code,
		)
	elif model_args.model_type == 'DNABERT-2':
		logger.info(f'Loading {model_args.model_type}')
		model = DNABERT2ForNucleotideLevel.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="regression",
			trust_remote_code=model_args.trust_remote_code,
		)
	elif model_args.model_type == 'GENA-LM':
		logger.info(f'Loading {model_args.model_type}')
		model = GENALMForNucleotideLevel.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="regression",
			trust_remote_code=model_args.trust_remote_code,
		)
	elif model_args.model_type == 'Caduceus':
		logger.info(f'Loading {model_args.model_type}')
		model = CaduceusForNucleotideLevel.from_pretrained(
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
		model = EsmForNucleotideLevel.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="regression",
			trust_remote_code=model_args.trust_remote_code,
		)
	elif model_args.model_type == 'Evo':
		logger.info(f'Loading {model_args.model_type}')
		model = StripedHyenaForNucleotideLevel.from_pretrained(
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
		model = HyenaDNAForNucleotideLevel.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			num_labels=train_dataset.num_labels,
			problem_type="regression",
			trust_remote_code=model_args.trust_remote_code,
		)
		training_args.save_safetensors = False  # Attention: The weights trying to be saved contained shared tensors, save model weight in *.bin

	early_stopping = EarlyStoppingCallback(early_stopping_patience=10)
             
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
   
   
# def test_dataset_loader(dataset, args):
# 	return DataLoader(
# 		dataset,
# 		batch_size=args.per_device_eval_batch_size,
# 		shuffle=False,
# 		collate_fn=test_data_collator,
# 		num_workers=4
# 	)
# test_data_loader1 = test_dataset_loader(public_test_dataset,training_args)
# test_data_loader2 = test_dataset_loader(private_test_dataset,training_args)
# make_pred_file(training_args, model, [test_data_loader1, test_data_loader2],postfix=training_args.output_dir.split('/')[-1])


#how to get score:
#submit to kaggle with command like
#kaggle competitions submit -c stanford-covid-vaccine -f xxx/submission_yyy.csv -m "Message"
