import json
import os
import shutil
from argparse import ArgumentParser
from glob import glob
import torch
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file
from utils import get_logger
from tqdm import tqdm
from typing import List
from fnmatch import fnmatch

from kernel import weight_quant

logger = get_logger()


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--input-bf16-hf-path", "-i", type=str, required=True)
    parser.add_argument("--output-fp8-hf-path", "-o", type=str, required=True)
    parser.add_argument("--nums-gpu", "-n", type=int, default=1)
    parser.add_argument("--scale", "-s", type=float, default=448.0)
    parser.add_argument("--block-size", "-bs", type=int, default=128)
    return parser.parse_args()


def proj2scale(name):
    return name + "_scale_inv"


def copy_files(source_dir, target_dir):
    exclude_file = ["*.safetensors", "*.bin", "model.safetensors.index.json", "config.json"]
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        if os.path.isfile(file_path):
            should_skip = False
            for pattern in exclude_file:
                if fnmatch(filename, pattern):
                    should_skip = True
                    break
            if not should_skip:
                target_path = os.path.join(target_dir, filename)
                shutil.copy2(file_path, target_path)
                logger.info(f"'{filename}' copyed and saved to '{target_path}'")


def update_config(model_index_file, config_file, bf16_path, block_size):
    if not os.path.exists(config_file):
        # copy from bf16 model
        bf16_config_file = os.path.join(bf16_path, "config.json")
        shutil.copy(bf16_config_file, config_file)

        # modify config.json and save it
        config = json.load(open(config_file))
        if "quantization_config" in config:
            quant_config = config["quantization_config"]
            quant_config["fmt"] = "e4m3"
            quant_config["quant_method"] = "fp8"
            quant_config["weight_block_size"] = [block_size, block_size]
            quant_config["activation_scheme"] = "dynamic"
        else:
            config["quantization_config"] = {
                "activation_scheme": "dynamic",
                "quant_method": "fp8",
                "fmt": "e4m3",
                "weight_block_size": [block_size, block_size],
            }
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)
        logger.info(f"config.json modified and saved to {config_file}")

    if not os.path.exists(model_index_file):
        # copy from bf16 model
        bf16_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
        shutil.copy(bf16_model_index_file, model_index_file)

    # add scale_inv
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    scale_map = {}
    scale_count = 0
    for k, v in weight_map.items():
        if "_proj" in k and not "eh_proj" in k and not "_scale_inv" in k:
            scale_name = proj2scale(k)
            scale_map[scale_name] = v
            scale_count += 1
    weight_map.update(scale_map)

    logger.info(f"scale_count: {scale_count}")

    with open(model_index_file, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)
    logger.info(
        f"'model.safetensors.index.json' modified and saved to '{model_index_file}'"
    )

    return weight_map, scale_count


if __name__ == "__main__":
    args = arg_parser()
    bf16_path, fp8_path, nums_gpu, block_size, scale = (
        args.input_bf16_hf_path,
        args.output_fp8_hf_path,
        args.nums_gpu,
        args.block_size,
        args.scale,
    )

    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(fp8_path, exist_ok=True)
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    config_file = os.path.join(fp8_path, "config.json")

    copy_files(bf16_path, fp8_path)
    weight_map, scale_count = update_config(
        model_index_file, config_file, bf16_path, block_size
    )

    safetensor_files = list(glob(os.path.join(bf16_path, "*.safetensors")))
    safetensor_files.sort()
    logger.info(f"number of safetensor files: {len(safetensor_files)}")

    # NOTE: 每个 GPU 处理一部分
    def process_data(sf_files: List[str], rank: int):
        cnt = 0
        tbar = tqdm(sf_files, position=rank)
        for safetensor_file in tbar:
            file_name = os.path.basename(safetensor_file)
            tbar.set_description(f"Processing {file_name}")
            device = f"cuda:{rank}"
            state_dict = load_file(safetensor_file, device=device)
            new_state_dict = {}
            for weight_name, weight in state_dict.items():
                scale_inv_name = f"{weight_name}_scale_inv"
                if scale_inv_name in weight_map:
                    assert weight.element_size() == 2
                    cnt += 1
                    fp8_weight, scale_inv = weight_quant(weight, block_size, scale)
                    new_state_dict[weight_name] = fp8_weight
                    new_state_dict[scale_inv_name] = scale_inv
                else:
                    new_state_dict[weight_name] = weight
            new_safetensor_file = os.path.join(fp8_path, file_name)
            save_file(new_state_dict, new_safetensor_file)
        return cnt

    with mp.Pool(processes=nums_gpu) as pool:
        quant_cnt = pool.starmap(  # 步长为 nums_gpu
            process_data, [(safetensor_files[x::nums_gpu], x) for x in range(nums_gpu)]
        )

    quant_count = sum(quant_cnt)
    logger.info(f"quant_count: {quant_count}, scale_count: {scale_count}")

    assert quant_count == scale_count
    logger.info(f"{quant_count} weights are quantized.")
    logger.info("done")
