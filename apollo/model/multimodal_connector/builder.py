import torch
import torch.nn as nn
from typing import Dict, List, Union
from .projectors import build_mm_projector
from ..utils import initialize_weights

from .perciver_idefics import Idefics2PerceiverResampler
from .flamino_perciver import FlamingoResampler
from .attentive_pooler import AttentivePooler
from .average_pooler import AveragePooler
from .perciver import PerceiverResampler
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, AutoModel
import torch.nn.functional as F
import json, os
#from transformers.modeling_utils import PreTrainedModel
#from transformers.configuration_utils import PretrainedConfig

class ConnectorConfig(PretrainedConfig):
    model_type = "mm_connector"
    def __init__(
        self,
        vision_hidden_size: List[int] = [],
        text_hidden_size: int = 0,
        num_patches: int = 24,
        rms_norm_eps: float = 1e-4, 
        token_input_shape: List[int] = [],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        self.num_patches = num_patches
        self.rms_norm_eps=rms_norm_eps
        self.token_input_shape = token_input_shape

    @classmethod
    def load_config(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "ConnectorConfig":
        cls._set_token_in_kwargs(kwargs)
        config_dict, kwargs = cls.get_config_from_json(pretrained_model_name_or_path, **kwargs)
        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def get_config_from_json(cls, config_file, **kwargs):
        with open(config_file, 'r') as file:
            config_data = json.load(file)
        return config_data, kwargs



class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x
    
class Interpolator(nn.Module):
    def __init__(self, output_size=(7, 7)):
        super().__init__()
        self.output_size = output_size
    def forward(self, x):
        return F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
    
class AdaptivePooler(nn.Module):
    def __init__(self, output_size=(1, 1)):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)
    def forward(self, x):
        return self.pool(x)

class ConvResize(nn.Module):
    def __init__(self, in_channels, out_channels, input_height, input_width, output_height, output_width):
        super(ConvResize, self).__init__()
        
        # Calculate the necessary kernel size, stride, and padding
        # These calculations assume that we want to use the same configuration for both dimensions
        self.stride_height = input_height // output_height
        self.stride_width = input_width // output_width
        
        self.kernel_height = input_height - (output_height - 1) * self.stride_height
        self.kernel_width = input_width - (output_width - 1) * self.stride_width
        
        # Define the convolutional layer
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=(self.kernel_height, self.kernel_width),
                              stride=(self.stride_height, self.stride_width))
        
    def forward(self, x):
        # Apply the convolutional layer
        x = self.conv(x)
        return x


class Connector(PreTrainedModel):
    config_class = ConnectorConfig
    def __init__(self, connector_config: ConnectorConfig) -> None:
        super().__init__(connector_config)
        self.connector_config = connector_config
        self.proj = build_mm_projector(connector_config.vision_hidden_size, connector_config.text_hidden_size, connector_config.projector_type, token_input_shape=connector_config.token_input_shape)
        self.resampler = self._build_resampler(connector_config.resampler_type, connector_config)
        if getattr(self.connector_config, 'initialize', False):
            self.resampler.initialize()
            self.apply(initialize_weights)

    def _build_resampler(self, resampler_type: str, config: ConnectorConfig):
        resampler_mapping = {
            'perciver': PerceiverResampler,
            'flamingo_perciver': FlamingoResampler,
            'idefics2_perciver': Idefics2PerceiverResampler,
            'att_pool': AttentivePooler,
            'avg': AveragePooler
        }
        return resampler_mapping.get(resampler_type, resampler_mapping['perciver'])(config)
    
    def forward(self, x):
        x = self.proj(x)
        x = self.resampler(x)
        return x


def build_mm_connector(model_type_or_path,  config, vision_hidden_size, token_input_shape, **kwargs):
    if config.resume_path:
        assert os.path.exists(model_type_or_path), f"Resume mm connector path {model_type_or_path} does not exist!"
        return Connector.from_pretrained(model_type_or_path, torch_dtype=eval(config.model_dtype))

    model_type_or_path = os.path.join("apollo/connector_configs", model_type_or_path+".json")
    connector_config = ConnectorConfig.load_config(model_type_or_path, 
                                                    vision_hidden_size=vision_hidden_size, 
                                                    text_hidden_size=config.hidden_size, 
                                                    rms_norm_eps=config.rms_norm_eps,
                                                    token_input_shape=token_input_shape,
                                                    intialize = True)
    connector = Connector(connector_config)
    return connector

AutoConfig.register("mm_connector", ConnectorConfig)
AutoModel.register(ConnectorConfig, Connector)