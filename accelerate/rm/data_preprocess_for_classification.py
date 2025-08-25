#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import psutil
import argparse
import re
import json
import logging
import math
import os
import csv
import random
import copy
import numpy as np
import pandas as pd
import time
from itertools import chain
from pathlib import Path
import logging
import datasets
import torch
from datasets import load_dataset, load_from_disk
from datasets import Dataset, DatasetDict
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.28.0.dev0")

logger = logging.getLogger(__file__)
logger.info("INFO")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )

    parser.add_argument(
        "--train_file_dir", type=str, default=None, help="A csv or a json file containing the training data dir."
    )

    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument(
        "--extern_eval_file",
        type=str,
        default=None,
        help="filter tids",
    )
    parser.add_argument(
        "--data_cache_dir",
        type=str,
        default=None,
        help=" ",
    )
    parser.add_argument(
        "--process_type",
        type=str,
        default="llama",
        help=" ",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help=" ",
    )

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

    return args


def get_label_list(raw_dataset, split="train"):
    """Get the list of labels from a multi-label dataset"""

    if isinstance(raw_dataset[split]["label"][0], list):
        label_list = [label for sample in raw_dataset[split]["label"] for label in sample]
        label_list = list(set(label_list))
    else:
        label_list = raw_dataset[split].unique("label")
    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]
    return label_list



def main():
    args = parse_args()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    # accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # logger.info(accelerator.state, main_process_only=False)
    # If passed along, set the training seed now.
    # if args.seed is not None:
    #     set_seed(args.seed)


    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            train_files = args.train_file.split(';')
            data_files["train"] = train_files
        if args.validation_file is not None:
            validation_files = args.validation_file.split(';')
            data_files["validation"] = validation_files
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, keep_default_na=False, **dataset_args)

        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )

    column_names = raw_datasets["train"].column_names
    if args.process_type == 'llama' or args.process_type == 'Yi' or args.process_type == 'qwen2':
        text_column_name = "text" if "text" in column_names else column_names[1]
    elif args.process_type == 'alpaca':
        instruction_column_name = "instruction" if "instruction" in column_names else column_names[-3]
        input_column_name = "input" if "input" in column_names else column_names[-2]
        output_column_name = "output" if "output" in column_names else column_names[-1]

    # 移除预留验证集中tid
    def example_filter(examples):
        keys = examples.keys()
        tids = examples['tid']
        results = {key: [] for key in keys}
        for idx, tid_str in enumerate(tids):
            tid = tid_str.split('-')[0]
            if tid in eval_tids_set:
                continue

            for key in keys:
                results[key].append(examples[key][idx])
        return results

    if args.extern_eval_file:
        eval_tids_set = set()
        csv_reader = csv.reader(open(args.extern_eval_file))
        for line in csv_reader:
            eval_tids_set.add(str(line[0]))

        raw_datasets['train'] = raw_datasets['train'].map(
            example_filter,
            batched=True,
            desc=f"filter train examples")

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=not args.use_slow_tokenizer,
        trust_remote_code=True)

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]

    is_multi_label = False
    if is_regression:
        label_list = None
        num_labels = 1
        # regession requires float as label type, let's cast it if needed
        for split in raw_datasets.keys():
            if raw_datasets[split].features["label"].dtype not in ["float32", "float64"]:
                logger.warning(
                    f"Label type for {split} set to float32, was {raw_datasets[split].features['label'].dtype}"
                )
                features = raw_datasets[split].features
                features.update({"label": Value("float32")})
                try:
                    raw_datasets[split] = raw_datasets[split].cast(features)
                except TypeError as error:
                    logger.error(
                        f"Unable to cast {split} set to float32, please check the labels are correct, or maybe try with --do_regression=False"
                    )
                    raise error

    else:  # classification
        if raw_datasets["train"].features["label"].dtype == "list":  # multi-label classification
            is_multi_label = True
            logger.info("Label type is list, doing multi-label classification")
        # Trying to find the number of labels in a multi-label classification task
        # We have to deal with common cases that labels appear in the training set but not in the validation/test set.
        # So we build the label list from the union of labels in train/val/test.
        label_list = get_label_list(raw_datasets, split="train")
        for split in ["validation", "test"]:
            if split in raw_datasets:
                val_or_test_labels = get_label_list(raw_datasets, split=split)
                diff = set(val_or_test_labels).difference(set(label_list))
                if len(diff) > 0:
                    # add the labels that appear in val/test but not in train, throw a warning
                    logger.warning(
                        f"Labels {diff} in {split} set but not in training set, adding them to the label list"
                    )
                    label_list += list(diff)
        # if label is -1, we throw a warning and remove it from the label list
        for label in label_list:
            if label == -1:
                logger.warning("Label -1 found in label list, removing it.")
                label_list.remove(label)

        label_list.sort()
        num_labels = len(label_list)
        if num_labels <= 1:
            raise ValueError("You need more than one label to do classification.")

    label_to_id = {v: i for i, v in enumerate(label_list)}

    def multi_labels_to_ids(labels):
        ids = [0.0] * len(label_to_id)  # BCELoss requires float as target type
        for label in labels:
            ids[label_to_id[label]] = 1.0
        return ids

    def tokenize_function(examples):
        # Tokenize the texts
        result = tokenizer(examples["content"], padding="max_length", max_length=block_size, truncation=True)
        # result = tokenizer(examples["content"], padding="max_length", max_length=block_size, truncation=True, padding_side='left')

        if label_to_id is not None and "label" in examples:
            if is_multi_label:
                result["label"] = [multi_labels_to_ids(l) for l in examples["label"]]
            else:
                result["label"] = [(label_to_id[str(l)] if l != -1 else -1) for l in examples["label"]]
        return result

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    print("==== DEBUG ====")
    print(raw_datasets)
    print(column_names)
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        # remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )
    print("==== DEBUG ====")
    print(tokenized_datasets)
    cache_datasets = tokenized_datasets

    train_dataset = cache_datasets["train"]
    eval_dataset = cache_datasets["validation"]

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 10):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    print("checking data ...")
    for index in range(2):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")


    if args.data_cache_dir:
        if not os.path.exists(args.data_cache_dir):
            os.makedirs(args.data_cache_dir)

        cache_datasets.save_to_disk(args.data_cache_dir)
        logger.info("data cache write to {} succeeded!\n preview:{}".format(args.data_cache_dir, cache_datasets))

        if args.sample_size > 0:
            sample_datasets = DatasetDict()
            sample_datasets['train'] = cache_datasets['train'].shuffle(seed=42).select(
                range(args.sample_size))
            sample_datasets['validation'] = cache_datasets['validation']

            data_sample_dir = '{}_sample'.format(args.data_cache_dir)
            if not os.path.exists(data_sample_dir):
                os.makedirs(data_sample_dir)

            sample_datasets.save_to_disk(data_sample_dir)
            logger.info("data sample write to {} succeeded!\n preview:{}".format(args.data_cache_dir, sample_datasets))

    else:
        logger.error("data_cache_dir do not found!")

if __name__ == "__main__":
    main()
