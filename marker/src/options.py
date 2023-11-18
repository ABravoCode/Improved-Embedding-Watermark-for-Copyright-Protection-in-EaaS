import os
import math
import json
import wandb
import random
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import partial
from typing import Tuple
from scipy import stats

import torch
from torch import nn
from torch.utils.data import DataLoader

import datasets
from datasets import load_dataset, DatasetDict
from dataset.utils import load_mind
import evaluate

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from transformers import (
    AutoConfig,
    AutoTokenizer,
    SchedulerType,
    DataCollatorWithPadding,
    default_data_collator,
    get_scheduler,
)
import hashlib

from dataset.emb_cache import load_gpt_embeds
from model.gpt_cls import GPTClassifierConfig, GPTClassifier
from model.copier.bert import BertForClassifyWithBackDoor
from trigger.base import BaseTriggerSelector
from utils import merge_flatten_metrics

import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
    )

    parser.add_argument(
        "--job_name", type=str, default=None, help="The job name used for wandb logging"
    )

    # GPT3 configuration
    parser.add_argument(
        "--gpt_emb_dim", type=int, default=1536, help="The embedding size of gpt3."
    )
    parser.add_argument(
        "--gpt_emb_train_file",
        type=str,
        default=None,
        help="The gpt3 embedding file of sst2 train set.",
    )
    parser.add_argument(
        "--gpt_emb_validation_file",
        type=str,
        default=None,
        help="The gpt3 embedding file of sst2 validation set.",
    )
    parser.add_argument(
        "--gpt_emb_test_file",
        type=str,
        default=None,
        help="The gpt3 embedding file of sst2 test set.",
    )

    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="The train file of mind train set.",
    )

    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="The validation file of mind train set.",
    )

    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="The test file of mind train set.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )

    # Trigger Selection
    parser.add_argument(
        "--trigger_seed", type=int, default=2022, help="The seed for trigger selector."
    )
    parser.add_argument(
        "--trigger_min_max_freq",
        nargs="+",
        type=float,
        default=None,
        help="The max and min frequency of selected triger tokens.",
    )
    parser.add_argument(
        "--selected_trigger_num",
        type=int,
        default=100,
        help="The maximum number of triggers in a sentence.",
    )
    parser.add_argument(
        "--max_trigger_num",
        type=int,
        default=100,
        help="The maximum number of triggers in a sentence.",
    )
    parser.add_argument(
        "--word_count_file",
        type=str,
        default=None,
        help="The preprocessed word count file to load. Compute word count from dataset if None.",
    )
    parser.add_argument(
        "--disable_pca_evaluate", action="store_true", help="Disable pca evaluate."
    )
    parser.add_argument(
        "--disable_training", action="store_true", help="Disable pca evaluate."
    )

    # Model Copy
    parser.add_argument(
        "--verify_dataset_size",
        type=int,
        default=20,
        help="The number of samples of verify dataset.",
    )
    parser.add_argument(
        "--transform_hidden_size",
        type=int,
        default=1536,
        help="The dimention of transform hidden layer.",
    )
    parser.add_argument(
        "--transform_dropout_rate",
        type=float,
        default=0.0,
        help="The dropout rate of transformation layer.",
    )
    parser.add_argument(
        "--copy_learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--copy_num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--copy_max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--copy_gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--copy_num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )

    # GPT3 Classifier Config
    parser.add_argument(
        "--cls_hidden_dim",
        type=int,
        default=None,
        help="The hidden dimention of gpt3 classifier.",
    )
    parser.add_argument(
        "--cls_dropout_rate",
        type=float,
        default=None,
        help="The dropout rate of gpt3 classifier.",
    )
    parser.add_argument(
        "--cls_learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--cls_num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--cls_max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--cls_gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--cls_num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )

    parser.add_argument(
        "--data_name", type=str, default="sst2", help="dataset name for training."
    )

    parser.add_argument(
        "--project_name", type=str, default=None, help="project name for training."
    )

    # advanced
    parser.add_argument(
        "--use_copy_target",
        type=bool,
        default=False,
        help="Switch to the advanced version of EmbMarker to defend against distance-invariant attacks.",
    )

    # visualization
    parser.add_argument(
        "--plot_sample_num",
        type=int,
        default=600,
        help="Sample a subset of examples for visualization to decrease the figure size.",
    )
    parser.add_argument(
        "--vis_method",
        type=str,
        default="pca",
        choices=["pca", "tsne"],
        help="Choose a dimension reduction algprithm to visualize embeddings. Only support pca and tsne now.",
    )

    args = parser.parse_args()

    return args
