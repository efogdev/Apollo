# This file is modified from https://github.com/haotian-liu/LLaVA/
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    PretrainedConfig,
    PreTrainedModel,
)
import torch
from apollo.model import *
from apollo.model.utils import is_mm_model
from apollo.constants import *

def load_pretrained_model(
    model_path,
    model_name,
    model_base=None,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    **kwargs,
):
    kwargs = {"device_map": device_map, **kwargs}
    #import ipdb; ipdb.set_trace()
    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    else:
        kwargs["torch_dtype"] = torch.float16

    if is_mm_model(model_path):
        # Load LLaVA model
        ## TODO @yunhao: mind fixing lora
        if "lora" in model_name.lower() and model_base is None:
            warnings.warn(
                "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged."
            )
        if "lora" in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(
                model_base, use_fast=False, legacy=False
            )
            print("Loading LLaVA from base model...")
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
            )
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(
                    torch.empty(
                        token_num, tokem_dim, device=model.device, dtype=model.dtype
                    )
                )
                model.model.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(
                        token_num, tokem_dim, device=model.device, dtype=model.dtype
                    )
                )

            print("Loading additional LLaVA weights...")
            if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
                non_lora_trainables = torch.load(
                    os.path.join(model_path, "non_lora_trainables.bin"),
                    map_location="cpu",
                )
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download

                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id, filename=filename, subfolder=subfolder
                    )
                    return torch.load(cache_file, map_location="cpu")

                non_lora_trainables = load_from_hf(
                    model_path, "non_lora_trainables.bin"
                )
            non_lora_trainables = {
                (k[11:] if k.startswith("base_model.") else k): v
                for k, v in non_lora_trainables.items()
            }
            if any(k.startswith("model.model.") for k in non_lora_trainables):
                non_lora_trainables = {
                    (k[6:] if k.startswith("model.") else k): v
                    for k, v in non_lora_trainables.items()
                }
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel

            print("Loading LoRA weights...")
            model = PeftModel.from_pretrained(model, model_path)
            print("Merging LoRA weights...")
            model = model.merge_and_unload()
            print("Model is loaded...")
        ## TODO @yunhao: mind fixing this
        elif model_base is not None:
            # this may be mm projector only
            print("Loading LLaVA from base model...")
            cfg_pretrained = AutoConfig.from_pretrained(
                model_path, trust_remote_code=True
            )
            mm_config_wrapper(config, kwargs)
            if "mpt" in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, "configuration_mpt.py")):
                    shutil.copyfile(
                        os.path.join(model_base, "configuration_mpt.py"),
                        os.path.join(model_path, "configuration_mpt.py"),
                    )
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                model = LlavaMPTForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_base, use_fast=False, legacy=False
                )
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                )
        else:
            config = AutoConfig.from_pretrained(model_path)
            config.resume_path = model_path
            prepare_config_for_eval(config, kwargs)
            model = ApolloForCausalLM(
                config=config,
                low_cpu_mem_usage=True,
                **kwargs
            )
            tokenizer = model.get_model().tokenizer
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, **kwargs
            )
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.float16)
        else:
            if "mpt" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, use_fast=False, legacy=False
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, **kwargs
                )
    model.eval()
    vision_processor = None
    if is_mm_model(model_path):
        use_mm_start_end = getattr(model.config, "use_mm_start_end", False)
        use_mm_patch_token = getattr(model.config, "use_mm_patch_token", True)
        if use_mm_patch_token:
            tokenizer.add_tokens(list(X_PATCH_TOKEN.values()), special_tokens=True)
        if use_mm_start_end:
            tokenizer.add_tokens(list(X_START_TOKEN.values())+list(X_END_TOKEN.values()), special_tokens=True)

        model.resize_token_embeddings(len(tokenizer))
        vision_tower = model.get_vision_tower()
        vision_tower.to(device=device, dtype=torch.float16)
        mm_connector = model.get_mm_connector()
        mm_connector.to(device=device, dtype=torch.float16)
        vision_processor = vision_tower.vision_processor

    if hasattr(model.get_llm().config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, vision_processor, context_len

def parse_model_name_or_path(config: PretrainedConfig, model_name="llm", suffix="_cfg"):
    target_model = f"{model_name}{suffix}"
    target_cfg = getattr(config, target_model, None)
    if isinstance(target_cfg, str):
        return target_cfg
    elif isinstance(target_cfg, dict):
        return target_cfg['vision_towers']
    else:
        raise ValueError(f"Invalid {target_model} configuration!")

def prepare_config_for_eval(config: PretrainedConfig, kwargs: dict):
    try:
        # compatible with deprecated config convention
        if getattr(config, "vision_tower_cfg", None) is None:
            config.vision_tower_cfg = config.mm_vision_tower
    except AttributeError:
        raise ValueError(f"Invalid configuration! Cannot find vision_tower in config:\n{config}")
    
    config.model_dtype = kwargs.pop("torch_dtype").__str__()
    # siglip does not support device_map = "auto"
    vision_tower_names = parse_model_name_or_path(config, "vision_tower")
    if len(vision_tower_names) > 0 and False:
        kwargs["device_map"] = "cuda"