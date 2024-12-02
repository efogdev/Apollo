from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..utils import initialize_weights
from .utils.activations import ACT2FN

from torch.nn.init import trunc_normal_, normal_
import torch.utils.checkpoint
from torch import nn
import torch
import math


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

def get_3d_sincos_pos_embed(
    embed_dim,
    grid_size,
    grid_depth,
    cls_token=False,
    uniform_power=False
):
    """
    grid_size: int of the grid height and width
    grid_depth: int of the grid depth
    returns:
        pos_embed: [grid_depth*grid_size*grid_size, embed_dim] (w/o cls_token)
                or [1+grid_depth*grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_h, grid_d, grid_w = np.meshgrid(grid_h, grid_d, grid_w)  # order of meshgrid is very important for indexing as [d,h,w]

    if not uniform_power:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim/6)*2)

    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)  # (T*H*W, D1)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)  # (T*H*W, D2)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)  # (T*H*W, D3)
    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    pos_embed = pos_embed[:, :embed_dim]
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed.reshape(grid_depth, grid_size, grid_size, embed_dim)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    returns:
        pos_embed: [grid_size*grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)  # order of meshgrid is very important for indexing as [h, w]

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)  # (H*W, D/2)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    embed_dim: output dimension for each position
    grid_size: int of the grid length
    returns:
        pos_embed: [grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size, embed_dim] (w/ cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    returns: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_abs_pos_3d(abs_pos, tgt_size):
    # abs_pos: [T, H, W, C]
    # tgt_size: M (total number of positions)
    # Returns: [M, C]

    src_T, src_H, src_W, C = abs_pos.shape
    tgt_T = int(round(M ** (1/3)))
    tgt_H = tgt_W = tgt_T  # Assuming cubic grid

    if (src_T, src_H, src_W) != (tgt_T, tgt_H, tgt_W):
        # Reshape to [1, C, src_T, src_H, src_W]
        abs_pos = abs_pos.transpose(3, 0, 1, 2)[None, ...]
        # Interpolate
        abs_pos = torch.nn.functional.interpolate(
            abs_pos.float(),
            size=(tgt_T, tgt_H, tgt_W),
            mode="trilinear",
            align_corners=False,
        )
        # Reshape back to [tgt_T, tgt_H, tgt_W, C]
        abs_pos = abs_pos[0].transpose(1, 2).transpose(2, 3).transpose(3, 0)
        abs_pos = abs_pos.reshape(-1, C)
    else:
        abs_pos = abs_pos.reshape(-1, C)

    return abs_pos


class WeightedNorm(nn.Module):
    def __init__(self, hidden_size):
        """
        WeightedNorm
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm(self.hidden_size)
        self.wheight = nn.Parameter(torch.ones(self.hidden_size))
        normal_(self.wheight, mean = 1, std=.02)

    def forward(self, x):
        x = self.norm(x)
        return x * self.wheight


class PerceiverMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        output_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, output_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)




class Idefics2PerceiverAttention(nn.Module):
    def __init__(self, connector_config, layer_idx: Optional[int] = None) -> None:
        """Perceiver Cross-Attention Module --> let long-form inputs be `context`, resampled embeddings be `latents`"""
        super().__init__()

        self.layer_idx = None
        self.hidden_size = connector_config.text_hidden_size
        self.num_heads = connector_config.resampler_n_heads
        self.head_dim = connector_config.resampler_head_dim
        self.num_key_value_heads = connector_config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.is_causal = False

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Runs Perceiver Self-Attention, with special (context, latents) appended along the `seq` dimension!

        Args:
            latents (`torch.Tensor`): Tensor of shape [bsz, n_latents, embed_dim] representing fixed length latents to compress to.
            context (`torch.Tensor`): Tensor of shape [bsz, seq, embed_dim] representing long-form context to resample.
            output_attentions (`bool`, *optional*, defaults to `False`): Whether to return attention weights.
            use_cache (`bool`, *optional*, defaults to `False`): Whether to use past_key_value for caching.
        """
        bsz, q_len, _ = latents.size()
        kv_seq_len = q_len + context.size()[1]

        hidden_states = torch.concat([context, latents], dim=-2)

        query_states = self.q_proj(latents)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value



IDEFICS2_PERCEIVER_ATTENTION_CLASSES = {
    "eager": Idefics2PerceiverAttention,
}


class Idefics2PerceiverLayer(nn.Module):
    def __init__(self, connector_config, layer_idx: int):
        super().__init__()
        self.hidden_size = connector_config.text_hidden_size
        self.n_latents = connector_config.num_output_tokens
        self.depth = connector_config.resampler_depth
        self.ff_multi = connector_config.ff_multi

        self.input_latents_norm = WeightedNorm(self.hidden_size)
        self.input_context_norm = WeightedNorm(self.hidden_size)
        self.self_attn = IDEFICS2_PERCEIVER_ATTENTION_CLASSES[connector_config._attn_implementation](connector_config, layer_idx=layer_idx)
        self.post_attention_layernorm = WeightedNorm(self.hidden_size)
        self.mlp = PerceiverMLP(
            hidden_size=connector_config.text_hidden_size,
            intermediate_size=connector_config.text_hidden_size * self.ff_multi,
            output_size=connector_config.text_hidden_size,
            hidden_act=connector_config.hidden_act,
        )


    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        pos_embed,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            latents (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            context (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        pos_embed_2 = get_abs_pos_3d(pos_embed, context.size(1))
        residual = latents

        latents = self.input_latents_norm(latents) 
        context = self.input_context_norm(context)
        
        latents, self_attn_weights, present_key_value = self.self_attn(
            latents=latents + pos_embed.unsqueeze(1),
            context=context + pos_embed_2.unsqueeze(1),
        )

        latents = residual + latents
        residual = latents

        latents = self.post_attention_layernorm(latents)
        latents = self.mlp(latents)
        latents = residual + latents

        outputs = (latents,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class PerceiverResampler(nn.Module):
    def __init__(self, connector_config) -> None:
        """
        Instantiates a Perceiver Resampler that operates over a sequence of embeddings (say from a ResNet or ViT or
        MAE) of a given dimension, performs `depth` blocks of cross-attention with a fixed `n_latents` inputs, then
        returns a Tensor of shape [bsz, n_latents, embed_dim]. The Resampler acts as a form of learned pooling and
        is derived from [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206).
        """
        super().__init__()
        self.hidden_size = connector_config.text_hidden_size
        self.hidden_act = connector_config.hidden_act
        self.n_latents = connector_config.num_output_tokens
        self.depth = connector_config.resampler_depth

        if (self.n_latents/2)**0.5 == int((self.n_latents/2)**0.5):
            grid_depth = 2
        elif (self.n_latents/3)**0.5 == int((self.n_latents/3)**0.5):
            grid_depth = 3
        else:
            grid_depth = 1
        
        grid_size = (self.n_latents / grid_depth) ** (1 / 2)
        
        assert grid_depth * grid_size**2 == self.n_latents, "number of latents doesn't match grid!!"
        self.pos_embed = nn.Parameter(torch.from_numpy(get_3d_sincos_pos_embed(self.hidden_size, grid_size, grid_depth)).float()).requires_grad_(False)
        self.grid_size=grid_size

        # Create Latents for Perceiver
        self.latents = nn.Parameter(torch.zeros(self.n_latents, self.hidden_size))

        # Create Transformer Blocks
        self.layers = nn.ModuleList([Idefics2PerceiverLayer(connector_config, idx) for idx in range(self.depth)])
        self.norm = WeightedNorm(self.hidden_size)

        self._use_flash_attention_2 = connector_config._attn_implementation == "flash_attention_2"
    
    def initialize(self):
        self.apply(initialize_weights)
        trunc_normal_(self.latents, mean = 0.5, std=.1)

    def forward(
        self,
        context: torch.Tensor,
        attention_mask: torch.Tensor = None, 
    ) -> torch.Tensor:
        # seq embed -> bsz seq embed
        
        latents = self.latents.unsqueeze(0).expand((context.shape[0], *self.latents.size()))  

        compressed_context = latents
        for i, perceiver_layer in enumerate(self.layers):
            layer_outputs = perceiver_layer(
                compressed_context,
                context,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                pos_embed = self.pos_embed,
            )
            compressed_context = layer_outputs[0]
    
        compressed_context = self.norm(compressed_context)
        return compressed_context
