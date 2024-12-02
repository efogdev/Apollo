import math
import warnings
import os, os.path as osp
import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
)
from apollo.utils import mprint
from .configs import *

def has_tokenizer(path):
    if (
        osp.exists(osp.join(path, "special_tokens_map.json"))
        and osp.exists(osp.join(path, "tokenizer_config.json"))
        and (osp.exists(osp.join(path, "tokenizer.model")) or osp.exists(osp.join(path, "tokenizer.json")))
    ):
        # print("[has_tokenizer]", path, True)
        return True
    from huggingface_hub import HfApi, file_exists
    from huggingface_hub.utils import validate_repo_id, HFValidationError
    api = HfApi()
    try:
        valid_hf_repo = api.repo_exists(path)
    except HFValidationError as e:
        valid_hf_repo = False
    if (
        valid_hf_repo
        and file_exists(path, "special_tokens_map.json")
        and file_exists(path, "tokenizer_config.json")
        and (file_exists(path, "tokenizer.model") or file_exists(path, "tokenizer.json"))
    ):
        return True
    return False


def context_length_extension(config):
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    model_max_length = getattr(config, "model_max_length", None)
    if orig_ctx_len and model_max_length > orig_ctx_len and False: #TODO: remove
        print(f"Scaling RoPE from {orig_ctx_len} to {model_max_length}")
        scaling_factor = float(math.ceil(model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    return config


def build_llm_and_tokenizer(
    model_name_or_path: str,
    config: PretrainedConfig,
    attn_implementation=None,
    model_max_length=None,
    *args,
    **kwargs,
) -> PreTrainedModel:
    llm_cfg = AutoConfig.from_pretrained(model_name_or_path)
    try:
        llm_arch = getattr(llm_cfg, "architectures")[0].lower()
    except BaseException:
        mprint(f'Cannot find LLM architecture, please check the "config.json" under "{llm_path}".')
    
    if 'apollo' in llm_arch:
        if "qwen" in llm_arch:
            llm_cfg = ApolloQwenConfig.from_pretrained(model_name_or_path)
        elif "llama" in llm_arch:
            llm_cfg = ApolloLlamaConfig.from_pretrained(model_name_or_path)
        try:
            llm_arch = getattr(llm_cfg, "architectures")[0].lower()
        except BaseException:
            mprint(f'Cannot find LLM architecture, please check the "config.json" under "{llm_path}".')
            
    if attn_implementation is not None:
        llm_cfg._attn_implementation = attn_implementation
    elif hasattr(config, 'attn_implementation'):
        llm_cfg._attn_implementation = config.attn_implementation 
    elif hasattr(config, '_attn_implementation'): 
        llm_cfg._attn_implementation = config._attn_implementation
    else: 
        mprint("attention implementation not defined for LLM")

    llm_cfg.model_max_length = model_max_length
    if model_max_length is not None:
        context_length_extension(llm_cfg)

    if 'apollo' not in llm_arch:
        llm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, config=llm_cfg, torch_dtype=eval(config.model_dtype), *args, **kwargs
        )
    
    llm_path = model_name_or_path
    if not has_tokenizer(llm_path):
        mprint("tokenizer found in VLM root folder. Move to ./{Apollo}/llm in the future.")
        #llm_path = osp.join(llm_path, "llm")
    
    # TODO(ligeng): use LLM class to judge to better compability.
    if "mpt" in model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path, 
            model_max_length=llm_cfg.model_max_length,
            padding_side="right",
        )
    elif "yi" in model_name_or_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            model_max_length=llm_cfg.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path, 
            model_max_length=llm_cfg.model_max_length,
            padding_side="right",
            use_fast=False,
            legacy=False,
        )
        
    # TODO(ligeng): is this necessary for llava?
    config.hidden_size = llm.config.hidden_size
    return llm, tokenizer