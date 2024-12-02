# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# This file is modified from https://github.com/haotian-liu/LLaVA/
import os, os.path as osp
from transformers import AutoConfig
from transformers import  PretrainedConfig
from huggingface_hub import snapshot_download, repo_exists
from huggingface_hub.utils import validate_repo_id, HFValidationError
from typing import Dict, List, Union
from torch import nn
import json


def get_model_config(config):
    default_keys = ["llm_cfg", "vision_tower_cfg", "mm_connector_cfg"]
    
    if hasattr(config, "_name_or_path") and len(config._name_or_path) >= 2:
        root_path = config._name_or_path
    else:
        root_path = config.resume_path 
        
    # download from huggingface
    if root_path is not None and not osp.exists(root_path):
        try:
            valid_hf_repo = repo_exists(root_path)
        except HFValidationError as e:
            valid_hf_repo = False
        if valid_hf_repo:
            root_path = snapshot_download(root_path)

    return_list = []
    for key in default_keys:
        cfg = getattr(config, key, None)
        if isinstance(cfg, dict):
            try:
                return_list.append(os.path.join(root_path, key[:-4]))
            except:
                raise ValueError(f"Cannot find resume path in config for {key}!")
        elif isinstance(cfg, PretrainedConfig):
            return_list.append(os.path.join(root_path, key[:-4]))
        elif isinstance(cfg, str):
            return_list.append(cfg)
        
    return return_list


def is_mm_model(model_path):
    """
    Check if the model at the given path is a visual language model.

    Args:
        model_path (str): The path to the model.

    Returns:
        bool: True if the model is an MM model, False otherwise.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    architectures = config.architectures
    if hasattr(config, "mm_model"):
        return config.mm_model
    elif hasattr(config, "vision_tower_cfg"):
        return True
    else:
        for architecture in architectures:
            if "apollo" in architecture.lower():
                return True
    return False


def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if "llava" in config and "llava" not in cfg.model_type:
        assert cfg.model_type == "llama"
        print(
            "You are using newer LLaVA code base, while the checkpoint of v0 is from older code base."
        )
        print(
            "You must upgrade the checkpoint to the new code base (this can be done automatically)."
        )
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = "LlavaLlamaForCausalLM"
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)




def init_a_linear(module, std):
    """Initialize a linear module with a specific standard deviation."""
    nn.init.normal_(module.weight, mean=0.0, std=std)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def initialize_weights(module):
    """Initialize the weights."""
    if 'MLP' in  module.__class__.__name__:
        for sub_module_name, sub_module in module.named_modules():
            if isinstance(sub_module, nn.Linear):
                fan_in = sub_module.weight.size(1)
                factor = 1.0
                if "down_proj" in sub_module_name:
                    factor = 2.0
                std = (0.4 / (fan_in * factor)) ** 0.5
                init_a_linear(sub_module, std)

    elif isinstance(module, nn.Linear):
        # He initialization for ReLU activations
        fan_in = module.weight.size(1)
        nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Conv2d):
        # He initialization for Conv2d with ReLU
        nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        # Standard initialization for LayerNorm
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
    elif isinstance(module, nn.Embedding):
        # Normal initialization for Embeddings
        nn.init.normal_(module.weight, mean=0, std=1)
    elif isinstance(module, nn.Parameter):
        print('initializing',module)
        # Xavier uniform initialization for Parameters
        nn.init.xavier_uniform_(module)