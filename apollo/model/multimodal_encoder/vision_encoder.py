# This file is modified from https://github.com/haotian-liu/LLaVA/

from abc import abstractmethod

import torch
import torch.nn as nn
from accelerate.hooks import add_hook_to_module
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from s2wrapper import forward as multiscale_forward
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

class VisionTowerConfig(PretrainedConfig):
    model_type = "vision_tower"
    def __init__(self, vision_tower_name: str = None, **kwargs):
        super().__init__()
        self.vision_tower_name = vision_tower_name


class VisionTower(PreTrainedModel):
    config_class = VisionTowerConfig
    
    def __init__(self, model_name_or_path: str, config: PretrainedConfig, vision_config: VisionTowerConfig = None):
        super().__init__(vision_config)
        self.vision_tower_name = model_name_or_path
        self.vision_config = vision_config
        self.select_layer = getattr(config, "mm_vision_select_layer", -2)
        self.select_feature = getattr(config, "mm_vision_select_feature", "patch")
        self.encode_batch_size = getattr(config, "encode_batch_size", 0) // 2
        self.num_encode_batch = getattr(config, "num_encode_batch", 0) // 2
        self.temporal_tubelet_size = getattr(vision_config, "tubelet_size", 1)

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def _forward(self, images, out_T=1):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_feature).to(image.dtype)
                image_feature = image_features.reshape(image_feature.shape[0], self.W, self.H, self.D)
                image_features.append(image_feature)
        else:
            original_shape = images.shape
            if len(original_shape) == 5 and self.T == 1:
                # downsample temporally if needed, and reshape from (B, T, C, W, H) to (B*T, C, W, H).
                images = images[:,::original_shape[1] // out_T,...]
                original_shape = images.shape
                images = images.view(-1, *original_shape[2:])
                
            image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_features).to(images.dtype)
            new_shape = list(image_features.shape[:-2]) + [self.W, self.H, self.hidden_size]
            image_features = image_features.reshape(new_shape)
            # Reshape back to (B, T, ...) if necessary
            if len(original_shape) == 5 and self.T == 1:
                # Assuming the feature dimension does not change, adapt the following line if it does
                feature_size = image_features.shape[1:]
                image_features = image_features.view(original_shape[0], original_shape[1], *feature_size)
                
        return image_features
    
    def forward(self, images):
        return self._forward(images)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device
        
    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    def _maybe_resize_pos_embeds(
        self,
        model: PreTrainedModel,
        image_processor: BaseImageProcessor,
        resolution: int = -1,
        interpolate_mode: str = "linear",
    ):
        if resolution in [model.config.image_size, -1]:
            return
        print(f"Resizing vision model's position embeddings to support higher vision resolution: from {model.config.image_size} to {resolution} ...")
        embeddings = model.vision_model.embeddings
        patch_size = embeddings.patch_size
        num_new_tokens = int((resolution // patch_size) ** 2)

        old_embeddings = embeddings.position_embedding
        match interpolate_mode:
            case "linear":
                ## Step 1: Calculate the corresponding patch ID (pid) in the current resolution (M patches) based on the target resolution (N patches). Formula: pid = pid / N * M
                ## Step 2:  Obtain new embeddings by interpolating between the embeddings of the two nearest calculated patch IDs. Formula: new_embeds = (pid - floor(pid)) * embeds[ceil(pid)] + (ceil(pid) - pid) * embeds[floor(pid)]
                import torch
                import torch.nn as nn

                if is_deepspeed_zero3_enabled():
                    import deepspeed

                    with deepspeed.zero.GatheredParameters([old_embeddings.weight], modifier_rank=None):
                        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
                else:
                    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
                new_embeddings = nn.Embedding(
                    num_new_tokens,
                    old_embedding_dim,
                    dtype=old_embeddings.weight.dtype,
                    device=old_embeddings.weight.device,
                )
                mapped_indices = (
                    torch.arange(num_new_tokens).to(old_embeddings.weight.device)
                    / (num_new_tokens - 1)
                    * (old_num_tokens - 1)
                )
                floor_indices = torch.clamp(mapped_indices.floor().long(), min=0, max=old_num_tokens - 1)
                ceil_indices = torch.clamp(mapped_indices.ceil().long(), min=0, max=old_num_tokens - 1)
                if is_deepspeed_zero3_enabled():
                    params = [old_embeddings.weight, new_embeddings.weight]
                    with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                        interpolated_embeds = (mapped_indices - floor_indices)[:, None] * old_embeddings.weight.data[
                            ceil_indices, :
                        ] + (ceil_indices - mapped_indices)[:, None] * old_embeddings.weight.data[floor_indices, :]
                else:
                    interpolated_embeds = (mapped_indices - floor_indices)[:, None] * old_embeddings.weight.data[
                        ceil_indices, :
                    ] + (ceil_indices - mapped_indices)[:, None] * old_embeddings.weight.data[floor_indices, :]
                new_embeddings.weight.data = interpolated_embeds
            case _:
                raise NotImplementedError

        if hasattr(old_embeddings, "_hf_hook"):
            hook = old_embeddings._hf_hook
            add_hook_to_module(new_embeddings, hook)
        new_embeddings.requires_grad_(old_embeddings.weight.requires_grad)
        ## update vision encoder's configurations
        model.config.image_size = resolution
        if hasattr(image_processor, "crop_size"):
            # CLIP vision tower
            image_processor.crop_size = resolution
        else:
            # SIGLIP vision tower
            assert hasattr(image_processor, "size")
            image_processor.size = {"height": resolution, "width": resolution}
        ## TODO define a '_reinitialize' method for VisionTower
        embeddings.position_embedding = new_embeddings
        embeddings.image_size = resolution
        embeddings.num_patches = embeddings.num_positions = num_new_tokens
        embeddings.position_ids = (
            torch.arange(embeddings.num_positions).expand((1, -1)).to(old_embeddings.weight.device)
        )

