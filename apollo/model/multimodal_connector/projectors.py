import torch
import torch.nn as nn
import re
from typing import Dict, List
from .utils.activations  import get_activation
from ..utils import initialize_weights
import einops


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class PerceiverMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        output_size: int,
        token_output_shape,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, output_size, bias=False)
        self.act_fn = get_activation(hidden_act)
        self.token_output_shape=token_output_shape
        self.apply(initialize_weights)

    def forward(self, x):
        x = x.view(x.shape[0], *self.token_output_shape, x.shape[-1])
        x = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return x

class ConvProj(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        output_size: int,
        kernel_size: int,
        hidden_act: str,
        token_output_shape: List[int] = [],
    ):
        super().__init__()
        self.token_input_shape=token_output_shape
        self.layer1 = nn.Conv2d(hidden_size, intermediate_size, kernel_size, kernel_size, bias=False)
        self.act_fn = get_activation(hidden_act)
        self.layer2 =  nn.Conv2d(intermediate_size, output_size, kernel_size, kernel_size, bias=False)
        self.apply(initialize_weights)

    def forward(self, x):
        original_shape = x.shape
        x = x.view(original_shape[0], *self.token_input_shape, original_shape[-1]).permute(0, 1, 4, 2, 3)
        reshaped_shape = x.shape
        if len(reshaped_shape) == 5:
            x = x.view(-1, *reshaped_shape[2:])

        x = self.layer2(self.act_fn(self.layer1(x)))
        
        if len(reshaped_shape) == 5:
            feature_size = x.shape[1:]
            x = x.view(reshaped_shape[0], reshaped_shape[1], *feature_size)

        return x.permute(0, 1, 3, 4, 2)


class STCConnector(nn.Module):
    def __init__(self, vision_hidden_size, text_hidden_size, token_output_shape: List[int] = [], downsample=(2, 2, 2), depth=4, mlp_depth=2):
        """Temporal Convolutional Vision-Language Connector.
        from LLaMA2: https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/main/videollama2/model/projector.py
        
        Args:
            config: config object.
            downsample: (temporal, height, width) downsample rate.
            depth: depth of the spatial interaction blocks.
            mlp_depth: depth of the vision-language projector layers.
        """
        super().__init__()
        self.token_input_shape=token_output_shape
        self.encoder_hidden_size = encoder_hidden_size = vision_hidden_size
        self.hidden_size = hidden_size = text_hidden_size
        self.output_hidden_size = output_hidden_size = text_hidden_size
        # TODO: make these as config arguments
        self.depth = depth
        self.mlp_depth = mlp_depth
        self.downsample = downsample
        from timm.models.regnet import RegStage
        from timm.models.layers import LayerNorm, LayerNorm2d
        if depth != 0:
            self.s1 = RegStage(
                depth=depth,
                in_chs=encoder_hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s1 = nn.Identity()
        self.sampler = nn.Sequential(
            nn.Conv3d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=downsample,
                stride=downsample,
                padding=1,
                bias=True
            ),
            nn.SiLU()
        )
        if depth != 0:
            self.s2 = RegStage(
                depth=depth,
                in_chs=hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s2 = nn.Identity()
            
        modules = [nn.Linear(hidden_size, output_hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(output_hidden_size, output_hidden_size))
        self.readout = nn.Sequential(*modules)

    def forward(self, x):
        """Aggregate tokens on the temporal and spatial dimensions.
        Args:
            x: input tokens [b, t, h, w, d] / [b, t, l, d]
        Returns:
            aggregated tokens [b, l, d]
        """
        original_shape = x.shape
        x = x.view(original_shape[0], *self.token_input_shape, original_shape[-1])
        reshaped_shape = x.shape
        t = x.size(1)
        if x.ndim == 4:
            hw = int(x.size(2) ** 0.5)
            x = einops.rearrange(x, "b t (h w) d -> b d t h w", h=hw, w=hw)
        elif x.ndim == 5:
            x = einops.rearrange(x, "b t h w d -> b d t h w")

        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        # 1. the first stage of the adapter
        x = self.s1(x)
        x = einops.rearrange(x, "(b t) d h w -> b d t h w", t=t)
        # 2. downsampler
        x = self.sampler(x)
        new_t, new_h, new_w = x.size(2), x.size(3), x.size(4)
        # 3. the second stage of the adapter
        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        x = self.s2(x)
        x = einops.rearrange(x, "(b t) d h w -> b (t h w) d", t=new_t)
        x = self.readout(x)
        x = einops.rearrange(x, "b (t h w) d -> b t h w d", t=new_t, h=new_h, w=new_w)
        return x



class STCConnectorV35(STCConnector):
    def __init__(self, vision_hidden_size, text_hidden_size, token_output_shape: List[int] = [], downsample=(2, 2, 2), depth=4, mlp_depth=2):
        super().__init__(vision_hidden_size=vision_hidden_size, text_hidden_size=text_hidden_size, token_output_shape=token_output_shape, downsample=downsample, depth=depth, mlp_depth=mlp_depth)
        self.sampler = nn.Sequential(
            nn.Conv3d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=downsample,
                stride=downsample,
                padding=0,
                bias=True
            ),
            nn.SiLU())




def build_mm_projector(input_dim, output_dim, projector_type, hidden_act='silu', delay_load=False, token_input_shape=0, **kwargs):
    if projector_type == 'perceiver_mlp':
        return PerceiverMLP(hidden_size=input_dim,
                            intermediate_size=output_dim * 4,
                            output_size=output_dim,
                            token_output_shape=token_input_shape,
                            hidden_act=hidden_act
                            )

    elif projector_type == 'stc_v35':
        return STCConnectorV35(input_dim, 
                               output_dim, 
                               token_output_shape=token_input_shape
                               )

    elif projector_type == 'stc':
        return STCConnector(input_dim, 
                            output_dim, 
                            token_output_shape=token_input_shape
                            )

    elif 'conv2d' in projector_type:
        kernel_size = re.match(r'.*conv2d_(\d+)x(\d+)$', projector_type)
        kernel_size = int(kernel_size.group(1))
        return ConvProj(hidden_size=input_dim,
                        intermediate_size=output_dim,
                        output_size=output_dim,
                        kernel_size=kernel_size,
                        hidden_act=hidden_act,
                        token_output_shape=token_input_shape
                        )
    else:
        modules = []
        modules.append(nn.Linear(input_dim, output_dim))
        mlp_gelu_match = re.match(r'.*mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match is not None:
            mlp_depth = int(mlp_gelu_match.group(1))
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(output_dim, output_dim))
    
        return nn.Sequential(*modules)        