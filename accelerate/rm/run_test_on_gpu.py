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
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import psutil
import argparse
import json
import logging
import math
import os
import csv
import random
import wandb
import time
from itertools import chain
from pathlib import Path

import datasets
import evaluate
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed, DummyOptim, DummyScheduler
from datasets import load_dataset, load_from_disk
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
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.28.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# wandb environ export
os.environ["WANDB_BASE_URL"] = "http://172.29.213.162:8900"
os.environ["WANDB_API_KEY"] = "local-7fdf70a47979d344ad6550e1032ba39b48cc110f"


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the cls task to train on.",
    )
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
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
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
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
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
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
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
        "--just_valid",
        action="store_true",
        help="Whether to only run for validing.",
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
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--extern_eval_file",
        type=str,
        default=None,
        help="filter tids",
    )
    parser.add_argument(
        "--extern_generate_file",
        type=str,
        default=None,
        help=" ",
    )
    parser.add_argument(
        "--deployed_ip_lists",
        type=str,
        default=None,
        help="deploy ips",
    )
    parser.add_argument(
        "--platform_eval_file",
        type=str,
        default=None,
        help="eval file name on the paofen platform",
    )
    parser.add_argument(
        "--init_generate",
        action="store_true",
        help=" ",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help=" ",
    )
    parser.add_argument(
        "--exp_iplist",
        type=str,
        default=None,
        help="exp ip list",
    )
    parser.add_argument(
        "--data_cache_dir",
        type=str,
        default=None,
        help=" ",
    )
    parser.add_argument(
        "--checkpointing_autosave_threshold",
        type=float,
        default=0.8,
        help=" ",
    )

    parser.add_argument(
        "--checkpointing_autosave_interval",
        type=int,
        default=3600,
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

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_classification_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        # accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # wandb init
    if accelerator.is_main_process and args.report_to == "wandb":
        wandb.login(key="local-7fdf70a47979d344ad6550e1032ba39b48cc110f")
        wandb.init(project=args.experiment_name)

    # Handle the repository creation
    # if accelerator.is_main_process:
    #     if args.push_to_hub:
    #         if args.hub_model_id is None:
    #             repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
    #         else:
    #             repo_name = args.hub_model_id
    #         create_repo(repo_name, exist_ok=True, token=args.hub_token)
    #         repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

    #         with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
    #             if "step_*" not in gitignore:
    #                 gitignore.write("step_*\n")
    #             if "epoch_*" not in gitignore:
    #                 gitignore.write("epoch_*\n")
    #     elif args.output_dir is not None:
    #         os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.data_cache_dir and not args.overwrite_cache:
        # read from local disk
        cache_datasets = load_from_disk(args.data_cache_dir)
        logger.info("load data from {} succeeded! \n preview:{}".format(args.data_cache_dir, cache_datasets))
    else:
        logger.warning("only supports data_cache")
        exit()

    train_dataset = cache_datasets["train"]
    eval_dataset = cache_datasets["validation"]

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = cache_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = cache_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = cache_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, num_labels=num_labels, trust_remote_code=True)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, trust_remote_code=True)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    # tokenizer = AutoTokenizer.from_pretrained(
    #     args.model_name_or_path, 
    #     use_fast=not args.use_slow_tokenizer, 
    #     trust_remote_code=True
    # )

    # import tokenization_yi
    # tokenizer = tokenization_yi.YiTokenizer(
    #     args.model_name_or_path,
    #     use_fast=not args.use_slow_tokenizer,
    #     trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=not args.use_slow_tokenizer)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = tokenizer.pad_token_id

    logger.info("config:{}".format(config))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        trust_remote_code=True
    )

    # model.load_state_dict(torch.load(args.model_name_or_path, map_location='cpu'))

    model.config.use_cache = False
    model.gradient_checkpointing_enable()


    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

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

    prompts = []
    if args.extern_generate_file:
        with open(args.extern_generate_file, 'r') as f:
            csv_reader = csv.reader(f, delimiter=",")
            header = next(csv_reader)
            i = 0
            for line in csv_reader:
                i = i + 1
                if len(line) == 2:
                    tid, prompt = line[0], line[1]
                else:
                    tid, course, prompt = line[0], line[1], line[2]
                prompt = prompt[0:block_size]
                prompts.append([tid, prompt])

    iplist = []
    ipcolumns = []
    if args.exp_iplist:
        with open(args.exp_iplist, 'r') as f:
            i = 0
            for line in f:
                iplist.append(line.rstrip())
                i = i + 1
                ipcolumns.append("ip" + str(i))

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

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: "
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()],
            "weight_decay": args.weight_decay,
        },
        # {
        #     "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        #     "weight_decay": args.weight_decay,
        # },
        # {
        #     "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        #     "weight_decay": 0.0,
        # },
    ]
    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    # Creates Dummy Optimizer if `optimizer` was spcified in the config file else creates Adam Optimizer
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
           or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # lr_scheduler = get_scheduler(
    #     name=args.lr_scheduler_type,
    #     optimizer=optimizer,
    #     num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
    #     num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    # )
    # Creates Dummy Scheduler if `scheduler` was spcified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (
            accelerator.state.deepspeed_plugin is None
            or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=args.max_train_steps, warmup_num_steps=args.num_warmup_steps
        )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    """
    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("cls_no_trainer", experiment_config)
        # accelerator.init_trackers(args.experiment_name, experiment_config)

    # exp iplist to wandb
    if accelerator.is_main_process and args.report_to == "wandb":
        if iplist:
            table = wandb.Table(data=[iplist], columns=ipcolumns)
            wandb.log({"iplist": table})
    """
    
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    min_step_loss = float('inf')
    last_save_time = time.time()


     
    with open(args.output_dir + "val_res.csv", "w") as fout:

    # if args.just_valid:
        print("run testing ...")
        # fout = open(os.path.join(args.output_dir, "val_res.csv"), "w")

        # model.eval()
        # metric = evaluate.load("/workspace/ningwang/scoring/metrics/accuracy")
        # metric_auc = evaluate.load("/workspace/ningwang/scoring/metrics/roc_auc")
        # metric_precision = evaluate.load("/workspace/ningwang/scoring/metrics/precision")
        # metric_recall = evaluate.load("/workspace/ningwang/scoring/metrics/recall")
        # metric_f1 = evaluate.load("/workspace/ningwang/scoring/metrics/f1")

        samples_seen = 0
        # total_cnt = 0
        # pos_cnt = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                # outputs = model(**batch)
                outputs = model(labels = batch['labels'], input_ids = batch['input_ids'], attention_mask = batch['attention_mask'])
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            logits = outputs.logits
            predictions, references, logits, tids = accelerator.gather((predictions, batch["labels"], logits, batch["tid"]))


            # logits_list = logits.tolist()
            # predictions_list = predictions.tolist()
            # labels_list = references.tolist()
            # tid_list = tids.tolist()
            # for idx in range(len(predictions_list)):
            #     p = logits_list[idx][0] / sum(logits_list[idx])
            #     q = logits_list[idx][1] / sum(logits_list[idx])
            #     print("%d,%d,%d,%.4f,%.4f,%.4f\n" % (tid_list[idx], predictions_list[idx], labels_list[idx], logits_list[idx][0], p, q))
            #     fout.write("%d,%d,%d,%.4f,%.4f,%.4f\n" % (tid_list[idx], predictions_list[idx], labels_list[idx], logits_list[idx][0], p, q))


            # diff = torch.nonzero(torch.ne(predictions, references)).flatten().tolist()

            # if diff:
            #     logits_list = logits.tolist()
            #     predictions_list = predictions.tolist()
            #     labels_list = references.tolist()
            #     tid_list = tids.tolist()
            #     for idx in diff:
            #         print("find badcase !!")
            #         p = logits_list[idx][0] / sum(logits_list[idx])
            #         q = logits_list[idx][1] / sum(logits_list[idx])
            #         print("%d,%d,%d,%.4f,%.4f,%.4f\n" % (tid_list[idx], predictions_list[idx], labels_list[idx], logits_list[idx][0], p, q))
            #         fout.write("%d,%d,%d,%.4f,%.4f,%.4f\n" % (tid_list[idx], predictions_list[idx], labels_list[idx], logits_list[idx][0], p, q))

            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]

            for tid, prediction, label, logits in zip(tids.tolist(), predictions.tolist(), references.tolist(), logits.tolist()):
                fout.write("%d,%d,%d,%.4f,%.4f\n" % (tid, prediction, label, logits[0], logits[1]))

            logger.info(f"samples_seen: {samples_seen}")
        #     metric.add_batch(
        #         predictions=predictions,
        #         references=references,
        #     )
        #     metric_auc.add_batch(
        #         prediction_scores=predictions,
        #         references=references,
        #     )
        #     metric_precision.add_batch(predictions=predictions, references=references)
        #     metric_recall.add_batch(predictions=predictions, references=references)
        #     metric_f1.add_batch(predictions=predictions, references=references)
        #     # print(f"accuracy: {eval_metric}", f"auc: {auc_metric}", f"f1: {metric_f1}")


        # eval_metric = metric.compute()
        # auc_metric = metric_auc.compute()
        # f1_metric = metric_f1.compute()
        # precision_metric = metric_precision.compute()
        # recall_metric = metric_recall.compute()
        # # logger.info(f"accuracy: {eval_metric}")
        # # logger.info(f"auc: {auc_metric}")
        # print("[FIANL RES]:", pos_cnt, total_cnt, "%.4f" % (pos_cnt / total_cnt))
        # print(f"accuracy: {eval_metric}", f"auc: {auc_metric}", f"f1: {f1_metric}",
        #       f"auc: {precision_metric}", f"f1: {recall_metric}")
        return 0

    

if __name__ == "__main__":
    main()
