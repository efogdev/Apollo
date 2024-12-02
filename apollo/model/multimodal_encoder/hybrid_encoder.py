import numpy as np
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import os
import torch.nn.functional as F
from .vision_encoder import VisionTower
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoConfig
from collections import OrderedDict


class HybridTowerConfig(PretrainedConfig):
    model_type = "hybrid_vision_tower"

    def __init__(self, configs=None, **kwargs):
        """
        Initializes the HybridTowerConfig.
        
        Args:
            configs (dict, optional): A dictionary where keys are component names and values are
                                      instances of configurations that have a `to_dict()` method.
            **kwargs: Additional keyword arguments that are passed to the superclass.
        """
        super().__init__(**kwargs)
        self.configs = {}
        
        if configs is not None:
            if not isinstance(configs, dict):
                raise TypeError("configs must be a dictionary where keys are component names and values are configuration objects.")
            
            for component_name, config in configs.items():
                if hasattr(config, 'to_dict'):
                    self.configs[component_name] = config.to_dict()
                else:
                    raise TypeError(f"The configuration for '{component_name}' does not have a to_dict() method and cannot be serialized.")
    
    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            dict: A dictionary containing all the keys and values of this configuration instance.
        """
        config_dict = super().to_dict()
        config_dict['configs'] = self.configs
        return config_dict


class HybridVisionTower(PreTrainedModel):
    def __init__(self, vision_towers: dict, configs: dict):
        composite_config = HybridTowerConfig(configs)
        super(HybridVisionTower, self).__init__(composite_config)
        self.vision_towers = list(vision_towers.keys())
        self._config = composite_config
        for vision_tower_name in self.vision_towers:
            setattr(self, vision_tower_name, vision_towers[vision_tower_name])

        self.vision_processor = [vision_towers[vt].vision_processor for vt in self.vision_towers]
        self.num_vision_encoders = len(vision_towers)
        self.W = self.H = max([vision_towers[vt].W for vt in self.vision_towers])
        self.T = max([vision_towers[vt].T for vt in self.vision_towers])
        self.max_tubelet_size = max([getattr(vision_towers[vt].vision_config, 'tubelet_size', 1) for vt in self.vision_towers])
        self._hidden_size = sum([vision_towers[vt].hidden_size for vt in self.vision_towers])
        self.token_output_shape = (self.T, self.W, self.H)
        self.config.num_vision_encoders = self.num_vision_encoders
        self.config.vision_towers = self.vision_towers
        self.config.token_output_shape = self.token_output_shape

        
    def forward(self, x):
        output_features = []
        for x_s, vision_tower_name in zip(x, self.vision_towers):
            vision_tower = getattr(self, vision_tower_name)
            features = vision_tower._forward(x_s, out_T=self.T)
            
            if len(features.shape) != len(self.token_output_shape) + 2:
                features = features.unsqueeze(1)

            if features.shape[-len(self.token_output_shape)-1:-1]!=self.token_output_shape:
                features = features.permute(0, 4, 1, 2, 3).contiguous()  # shape [B, D, T, W, H]
                features = F.interpolate(features.to(torch.float32), size=self.token_output_shape, mode='trilinear', align_corners=False).to(features.dtype)
                features = features.permute(0, 2, 3, 4, 1).contiguous()
            
            output_features.append(features)
        
        output_features = torch.cat(output_features, dim=-1)
        output_features = torch.flatten(output_features, start_dim=1, end_dim=-2)
        return output_features

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        state_dict = None,
        **kwargs,
    ):  
        if state_dict is None:
            state_dict  = self.state_dict()
            
        for vision_tower_name in self.vision_towers:
            vision_tower = getattr(self, vision_tower_name)
            vision_tower_state_dict = OrderedDict(
                {k.split(f"vision_tower.{vision_tower_name}.vision_tower.")[-1]: v for k, v in state_dict.items() if vision_tower_name in k}
            )
            vision_tower.vision_tower.save_pretrained(os.path.join(save_directory, vision_tower_name), state_dict=vision_tower_state_dict, **kwargs)
            vision_tower.vision_processor.save_pretrained(os.path.join(save_directory, vision_tower_name))
            
        config = self.config
        config.configs = {}
        config.save_pretrained(save_directory)

    @property
    def patch_size(self):
        return self._patch_size

    @property
    def image_size(self):
        return self._image_size

    @property
    def dtype(self):
        # Dynamically infer the dtype from the first parameter, if not explicitly specified
        if hasattr(self.vision_tower_1, 'dtype'):
            return self.vision_tower_1.dtype
        else:
            params = list(self.vision_tower_1.parameters())
            return params[0].dtype if len(params) > 0 else torch.float32  # Default to torch.float32 if no parameters

    @property
    def device(self):
        # Dynamically infer the device from the first parameter, if not explicitly specified
        if hasattr(self.vision_tower_1, 'device'):
            return self.vision_tower_1.device
        else:
            params = list(self.vision_tower_1.parameters())
            return params[0].device if len(params) > 0 else torch.device("cpu")  # Default to CPU if no parameters
    
    @property
    def hidden_size(self):
        return self._hidden_size


AutoConfig.register("hybrid_vision_tower", HybridTowerConfig)