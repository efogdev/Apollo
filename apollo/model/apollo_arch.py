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

import os, os.path as osp
import warnings
from abc import ABC, abstractmethod

import torch, logging

from transformers import (
    AutoConfig,
    PreTrainedModel,
)

from apollo.constants import (
    IGNORE_INDEX,
    X_TOKEN_INDEX,
    X_TOKEN,
    X_PATCH_TOKEN,
    X_START_TOKEN,
    X_END_TOKEN,
)


from collections import OrderedDict
from apollo.utils import mprint
from apollo.model.utils import get_model_config
from apollo.model.language_model.builder import build_llm_and_tokenizer
from apollo.model.multimodal_encoder.builder import build_vision_tower
from apollo.model.multimodal_connector.builder import build_mm_connector
from apollo.model.configuration_apollo import ApolloConfig
from transformers.modeling_utils import ContextManagers, no_init_weights


## TODO decide whether should we use metaclass
class ApolloMetaModel(ABC):
    def init_vlm(self, config: PreTrainedModel = None, *args, **kwargs):
        # TODO(ligeng): figure out how from_config and from_pretrained works in HF implementation.
        if hasattr(self, "llm") or hasattr(self, "vision_tower")  or hasattr(self, "mm_connector"):
            # already initialized, skipped
            return
        
        model_dtype = getattr(config, "model_dtype", "torch.float16")
        if not hasattr(config, "model_dtype"):
            warnings.warn("model_dtype not found in config, defaulting to torch.float16.")
            config.model_dtype = model_dtype
        
        # print("init_vlm(): config", config); input("DEBUG init_vlm")
        cfgs = get_model_config(config)
        if len(cfgs) == 3:
            llm_cfg, vision_tower_cfg, mm_connector_cfg = cfgs
        else:
            raise ValueError("`llm_cfg` `mm_connector_cfg` `vision_tower_cfg` not found in the config.")
        # print("init_vlm():", cfgs); input("DEBUG init_vlm")
        # print(llm_cfg, vision_tower_cfg, mm_connector_cfg); input("DEBUG init_vlm")
        #import ipdb; ipdb.set_trace()
        self.vision_tower = build_vision_tower(vision_tower_cfg, config)
        self.mm_connector = build_mm_connector(mm_connector_cfg, config, self.vision_tower.hidden_size, token_input_shape =self.vision_tower.config.token_output_shape)
        self.llm, self.tokenizer = build_llm_and_tokenizer(llm_cfg, config, *args, **kwargs)
        self.post_config()
        self.is_loaded = True

        assert (
            self.llm is not None or self.vision_tower is not None or self.mm_connector is not None
        ), "At least one of the components must be instantiated."
    
    @classmethod
    def load_from_config(cls, model_path_or_config, *args, **kwargs):
        pass
    
    def get_llm(self):
        llm = getattr(self, "llm", None)
        if type(llm) is list:
            llm = llm[0]
        return llm

    def get_lm_head(self):
        lm_head = getattr(self.get_llm(), "lm_head", None)
        return lm_head

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_mm_connector(self):
        mm_connector = getattr(self, "mm_connector", None)
        if type(mm_connector) is list:
            mm_connector = mm_connector[0]
        return mm_connector

    def get_temporal_embeddings(self):
        temporal_embeddings = getattr(self, 'temporal_embeddings', None)
        if type(temporal_embeddings) is list:
            temporal_embeddings = temporal_embeddings[0]
        return temporal_embeddings

    def get_1d_sincos_pos_embed_from_grid(self, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        returns: (M, D)
        """
        assert self.config.hidden_size % 2 == 0
        omega = np.arange(self.config.hidden_size // 2, dtype=float)
        omega /= self.config.hidden_size / 2.
        omega = 1. / 10000 ** omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    def post_config(self):
        self.training = self.get_llm().training
        self.config.llm_cfg = self.llm.config
        self.config.vision_tower_cfg = self.vision_tower.config
        self.config.mm_connector_cfg = self.mm_connector.config


class ApolloMetaForCausalLM(ABC):
    """This class is originally implemented by the LLaVA team and
    modified by Haotian Tang and Jason Lu based on Ji Lin's implementation
    to support multiple images and input packing."""
    @abstractmethod
    def get_model(self):
        pass

    ## FIXME we will use this function to load model in the future
    @classmethod
    def load_pretrained(cls, model_path_or_config, *args, **kwargs):
        kwargs.pop("config", None)
        if isinstance(model_path_or_config, str):
            config = AutoConfig.from_pretrained(model_path_or_config)
        elif isinstance(model_path_or_config, ApolloConfig):
            config = model_path_or_config
        else:
            raise NotImplementedError(f"wrong type, {type(model_path_or_config)} \
                                      {isinstance(model_path_or_config, ApolloConfig)}")

        model_dtype = getattr(config, "model_dtype", "torch.float16")
        if not hasattr(config, "model_dtype"):
            warnings.warn("model_dtype not found in config, defaulting to torch.float16.")
            config.model_dtype = model_dtype
        
        cfgs = get_model_config(config)
        if len(cfgs) == 3:
            llm_cfg, vision_tower_cfg, mm_connector_cfg = cfgs
        else:
            raise ValueError("`llm_cfg` `mm_connector_cfg` `vision_tower_cfg` not found in the config.")

        # print(llm_cfg, vision_tower_cfg, mm_connector_cfg); input("DEBUG load_pretrained")
        with ContextManagers([no_init_weights(_enable=True),]):
            vlm = cls(config, *args, **kwargs)
        # print(llm_cfg, vision_tower_cfg, mm_connector_cfg); input("DEBUG load_pretrained finish")
        
        if hasattr(vlm, "llm") or hasattr(vlm, "vision_tower")  or hasattr(vlm, "mm_connector"):
            if vlm.is_loaded:
                return vlm
        
        vlm.llm, vlm.tokenizer = build_llm_and_tokenizer(llm_cfg, config, *args, **kwargs)
        vlm.vision_tower = build_vision_tower(vision_tower_cfg, config)
        vlm.mm_connector = build_mm_connector(mm_connector_cfg, config, vlm.vision_tower.hidden_size, token_input_shape =vlm.vision_tower.config.token_output_shape)

        vlm.post_config()
        vlm.is_loaded = True

        # FIXME(ligeng, yunhao): llm should never be none here.
        assert (
            vlm.llm is not None or vlm.vision_tower is not None or vlm.mm_connector is not None
        ), "At least one of the components must be instantiated."
        return vlm
    
    ## FIXME we will use this function to save the model in the future
    def save_pretrained(self, output_dir, state_dict=None):
        if state_dict is None:
            # other wise fetch from deepspeed
            # state_dict = accelerator.get_state_dict(is_deepspeed_enabled)
            state_dict = self.state_dict()
        
        if getattr(self.get_model(), "tokenizer", None):
            self.get_model().tokenizer.save_pretrained(osp.join(output_dir, "llm"))

        if self.get_llm():
            mprint(f"saving llm to {osp.join(output_dir, 'llm')}")
            self.get_llm().config._name_or_path = osp.join(output_dir, "llm")
            llm_state_dict = OrderedDict({k.split("llm.")[-1]: v for k, v in state_dict.items() if "llm" in k})
            self.get_llm().save_pretrained(os.path.join(output_dir, "llm"), state_dict=llm_state_dict)
            self.config.llm_cfg = self.get_llm().config

        if self.get_vision_tower() and "radio" not in self.get_vision_tower().__class__.__name__.lower():
            mprint(f"saving vision_tower to {osp.join(output_dir, 'vision_tower')}")
            self.get_vision_tower().config._name_or_path = osp.join(output_dir, "vision_tower")
            vision_tower_state_dict = OrderedDict({k: v for k, v in state_dict.items() if "vision_tower" in k})
            self.get_vision_tower().save_pretrained(os.path.join(output_dir, "vision_tower"), state_dict=vision_tower_state_dict,)
            
            self.config.vision_tower_cfg = self.get_vision_tower().config
            if hasattr(self.config.vision_tower_cfg, 'auto_map'):
                delattr(self.config.vision_tower_cfg, 'auto_map')

        if self.get_mm_connector():
            mprint(f"saving mm_connector to {osp.join(output_dir, 'mm_connector')}")
            self.get_mm_connector().config._name_or_path = osp.join(output_dir, "mm_connector")
            mm_connector_state_dict = OrderedDict(
                {k.split("mm_connector.")[-1]: v for k, v in state_dict.items() if "mm_connector" in k}
            )
            self.get_mm_connector().save_pretrained(
                os.path.join(output_dir, "mm_connector"),
                state_dict=mm_connector_state_dict,
            )
            self.config.mm_connector_cfg = self.get_mm_connector().config
        ## update and save top-level config
        self.config._name_or_path = output_dir
        self.config.architectures = [self.__class__.__name__]
        self.config.save_pretrained(output_dir)

    def get_llm(self):
        return self.get_model().get_llm()
    
    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def get_mm_connector(self):
        return self.get_model().get_mm_connector()
    
    def get_input_embeddings(self):
        return self.get_model().get_llm().get_input_embeddings()

    def get_output_embeddings(self):
        return self.get_model().get_llm().get_output_embeddings()
    
    def resize_token_embeddings(self, embed_size):
        return self.get_model().get_llm().resize_token_embeddings(embed_size)
        
    def freezed_module_patch(self):
        '''
        Huggingface will call model.train() at each training_step. To ensure the expected behaviors for modules like dropout, batchnorm, etc., we need to call model.eval() for the freezed modules.
        '''
        if self.training:
            if self.get_llm() and not getattr(self.config, "tune_language_model", False):
                pass
                #logging.warning("Caution: Your LLM is currently in training mode, ensuring accurate gradient computation. Please be vigilant, particularly regarding BatchNorm and Dropout operations.")
            if self.get_vision_tower() and not getattr(self.config, "tune_vision_tower", False):
                self.get_vision_tower().eval()
            if self.get_mm_connector() and not getattr(self.config, "tune_mm_connector", False):
                self.get_mm_connector().eval()

    def _encode_mm(self, x):
        x = self.get_model().get_vision_tower()(x)
        x = self.get_model().mm_connector(x)
        return x

    def encode_mm(self, x):
        split_sizes = [x_s[0].shape[0] for x_s in x]
        x = [torch.cat([x_s[i] for x_s in x], dim=0) for i in range(self.get_model().get_vision_tower().num_vision_encoders)]
        x = self._encode_mm(x)
        assert not torch.isnan(x).any(), "nans found in mm features"
        x = torch.split(x, split_sizes, dim=0)
        #if self.config.config.add_temporal_sin:
        #   temporal_embeddings = self.get_model().get_temporal_embeddings()
        #   for xx in x:
        #       xx += temporal_embeddings.to(xx.device)[:len(xx)]
        return [xx.contiguous().view(-1, xx.shape[2]) for xx in x]

    def encode_mm_minibatch(self, x):
        split_sizes = [x_s[0].shape[0] for x_s in x]
        x = [torch.split(torch.cat([x_s[i] for x_s in x], dim=0), self.config.encode_batch_size) for i in range(self.get_model().get_vision_tower().num_vision_encoders)]
        swapped_x = []
        for i in range(len(x[0])):
            swapped_x.append([x_s[i] for x_s in x])

        features = []
        for xx in swapped_x:
            xx = self._encode_mm(xx)
            features.append(xx)
        x = torch.cat(features, dim=0)
        x = torch.split(x, split_sizes, dim=0)
        #if self.config.config.add_temporal_sin:
        #   temporal_embeddings = self.get_model().get_temporal_embeddings()
        #   for xx in x:
        #       xx += temporal_embeddings.to(xx.device)[:len(xx)]
        return [xx.contiguous().view(-1, xx.shape[2]) for xx in x]

    def encode_mm_fixed_passes(self, x):
        mm_feats = []
        for xx in x:
            num_mm = xx[0].size(0)
            num_mm_per_batch = (num_mm + self.config.num_encode_batch - 1) // self.config.num_encode_batch
            mm_feat = []
            start_idx = 0
            for _ in range(self.config.num_encode_batch):
                end_idx = min(start_idx + num_mm_per_batch, num_mm)
                mm = [tmp[start_idx:end_idx] for tmp in xx]
                mm = self._encode_mm(mm)
                mm_feat.append(mm)

            xx = torch.cat(mm_feat, dim=0)[:num_mm]
            xx = xx.contiguous().view(-1, xx.shape[2])
            #if self.config.config.add_temporal_sin:
            #   temporal_embeddings = self.get_model().get_temporal_embeddings()
            #   xx += temporal_embeddings.to(xx.device)[:len(xx)]
            mm_feats.append(xx)

        return mm_feats
    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, vision_input, data_types
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or vision_input is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and vision_input is not None
                and input_ids.shape[1] == 1
            ):
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (
                                attention_mask.shape[0],
                                target_shape - attention_mask.shape[1],
                            ),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )

        '''
            vision_input is a list of tuples, and data_type is a list of strings:
            data_type = ['image', 'video', 'video'..., 'text']
            (for one video and two image encoders)
            vision_input = 
            [
                [image(1, T, C, H, W), image(1, T, C, H, W), image(1, T, C, H, W)],
                [video(Nc1, C, T, H, W), video(Nc1, T, C, H, W), video(Nc1, T, C, H, W)],
                [video(Nc2, C, T, H, W), video(Nc2, T, C, H, W), video(Nc2, T, C, H, W)],
            ]
            -> video encoders typlically expect (C,T,H,W), images expect (C,H,W).
        '''
        # ====================================================================================================
        if getattr(self.config, "num_encode_batch", 0) > 0:
            merged_mm_features = self.encode_mm_fixed_passes(vision_input)
        elif getattr(self.config, "encode_batch_size", 0) > 0:
            merged_mm_features = self.encode_mm_minibatch(vision_input)
        else:
            merged_mm_features = self.encode_mm(vision_input)
            
        if not getattr(self.config, "tune_language_model", True) and getattr(self.config, "use_mm_start_end", False):
            raise NotImplementedError
        # ====================================================================================================
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask
        input_ids_copy = input_ids.clone()
        # kentang-mit@: Otherwise tokenizer out of bounds. Embeddings of image tokens will not be used.
        input_ids_copy[input_ids_copy == X_TOKEN_INDEX] = 0
        input_embeds = self.get_llm().model.embed_tokens(input_ids_copy)

        input_ids = [
            cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        input_embeds_1 = [
            cur_input_embeds[cur_attention_mask]
            for cur_input_embeds, cur_attention_mask in zip(input_embeds, attention_mask)
        ]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        #input_ids, new_input_embeds = self.inputs_merger(input_ids, input_embeds_1, merged_mm_features)
        new_labels = []
        new_input_embeds = []
        # print("BEFORE BATCH LOOP:", len(input_ids), input_ids[0].shape, input_ids[0].device, [(x == X_TOKEN_INDEX).sum() for x in input_ids])
        # kentang-mit@: If some part of the model is executed in the loop, the the loop length needs to be a constant.
        for batch_idx, (cur_labels, cur_input_ids, mm_features) in enumerate(zip(labels, input_ids, merged_mm_features)):
            cur_input_ids = input_ids[batch_idx]
            num_mm = (cur_input_ids == X_TOKEN_INDEX).sum()
            if num_mm == 0:
                cur_input_embeds_1 = input_embeds_1[batch_idx]
                cur_input_embeds = torch.cat([cur_input_embeds_1, mm_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(cur_labels)
                # kenang-mit@: we do not have placeholdr image for text-only data now.
                # cur_image_idx += 1
                continue
            
            if mm_features.shape[0] != num_mm:
                print(data_types[batch_idx])
                assert num_mm==len(mm_features), f'Error in {data_types[batch_idx]}{num_mm}=/={len(mm_features)} not the same number of vision tokens in and vision embeddings!'

            cur_input_embeds = input_embeds_1[batch_idx]
            image_token_indices = (
                [-1] + torch.where(cur_input_ids == X_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            cur_input_embeds_no_im = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_input_embeds_no_im.append(cur_input_embeds[image_token_indices[i] + 1 : image_token_indices[i + 1]])

            # cur_input_embeds = self.get_llm().embed_tokens(torch.cat(cur_input_ids_noim))
            # cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            for i in range(num_mm + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                #print("cur_new_input_embeds1", cur_new_input_embeds.shape[-1])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_mm:
                    cur_image_features = mm_features[i:i+1]
                    cur_new_input_embeds.append(cur_image_features)
                    #print("cur_new_input_embeds2", cur_new_input_embeds.shape[-1])
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
        
        
        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.get_llm().config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            if any(len(x) > tokenizer_model_max_length for x in new_input_embeds):
                warnings.warn("Inputs truncated!")
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.get_llm().config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )


    def inputs_merger(self, input_ids, input_embeds, merged_mm_features):
        ## NOTE: does not support zero3/multinode
        new_input_embeds = []
        # kentang-mit@: If some part of the model is executed in the loop, the the loop length needs to be a constant.
        for batch_idx, (cur_input_ids, cur_input_embeds, mm_features) in enumerate(zip(input_ids, input_embeds, merged_mm_features)):
            num_mm = (cur_input_ids == X_TOKEN_INDEX).sum()
            if num_mm == 0:
                cur_input_embeds = torch.cat([cur_input_embeds, mm_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                # kenang-mit@: we do not have placeholdr image for text-only data now.
                # cur_image_idx += 1
                continue

            image_token_indices = (
                [-1] + torch.where(cur_input_ids == X_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            )
            cur_input_embeds_no_im = []
            for i in range(len(image_token_indices) - 1):
                cur_input_embeds_no_im.append(cur_input_embeds[image_token_indices[i] + 1 : image_token_indices[i + 1]])

            cur_new_input_embeds = []
            for i in range(num_mm + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                if i < num_mm:
                    cur_image_features = mm_features[i:i+1]
                    cur_new_input_embeds.append(cur_image_features)

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            new_input_embeds.append(cur_new_input_embeds)
        return input_ids, new_input_embeds

    def inputs_merger_inplace(self, input_ids, input_embeds, merged_mm_features):
        ## NOTE: does not support zero3/multinode
        new_input_embeds = []
        for batch_idx, (cur_input_ids, cur_input_embeds, mm_features) in enumerate(zip( input_ids, input_embeds, merged_mm_features)):
            num_mm = sum(cur_input_ids == X_TOKEN_INDEX)  ## number of mm tokens in batch
            if num_mm == 0:
                cur_input_embeds = torch.cat([cur_input_embeds, mm_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                continue
        # ====================================================================================================
            # Replace the embeddings at MM token positions with MM features
            if mm_features.shape[0] != num_mm:
                raise ValueError(f"Mismatch in number of MM features {num_mm} and MM tokens {mm_features.shape[0]}")

            # Mask where MM tokens are present
            mm_mask = cur_input_ids == X_TOKEN_INDEX
            #cur_input_embeds = self.get_model().embed_tokens(cur_input_ids * ~mm_mask)

            cur_input_embeds = cur_input_embeds.clone()
            cur_input_embeds[mm_mask] = mm_features
            new_input_embeds.append(cur_input_embeds)
            
        return input_ids, new_input_embeds

    def repack_multimodal_data(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        inputs_embeds,
        labels,
    ):
        # kentang-mit@: reorder and repack (reduce computation overhead)
        # requires transformers replacement.
        new_inputs_embeds = []
        new_position_ids = []
        new_labels = []
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        sorted_seqlens_in_batch, sorted_idx = torch.sort(seqlens_in_batch, descending=True)
        # print(sorted_seqlens_in_batch)
        max_seqlen = inputs_embeds.shape[1]

        cur_inputs_embeds = []
        cur_position_ids = []
        cur_labels = []
        cur_batch_len = 0
        # print(sorted_seqlens_in_batch.device, len(sorted_seqlens_in_batch), max_seqlen)
        for i in range(len(sorted_seqlens_in_batch)):
            cur_seqlen = sorted_seqlens_in_batch[i].item()
            if cur_seqlen + cur_batch_len <= max_seqlen:
                cur_batch_len += cur_seqlen
                # each item: num_tokens x num_channels
                # remove padding on-the-fly
                cur_inputs_embeds.append(inputs_embeds[sorted_idx[i]][attention_mask[sorted_idx[i]]])
                # each item: num_tokens
                cur_position_ids.append(
                    torch.arange(
                        cur_inputs_embeds[-1].shape[0],
                        device=cur_inputs_embeds[-1].device,
                    )
                )
                # each item: num_tokens
                # remove padding on-the-fly
                cur_labels.append(labels[sorted_idx[i]][attention_mask[sorted_idx[i]]])
            else:
                new_inputs_embeds.append(torch.cat(cur_inputs_embeds, 0))
                new_position_ids.append(torch.cat(cur_position_ids, 0))
                new_labels.append(torch.cat(cur_labels, 0))
                # The current batch is too long. We will start a new batch.
                cur_batch_len = cur_seqlen
                cur_inputs_embeds = [inputs_embeds[sorted_idx[i]][attention_mask[sorted_idx[i]]]]
                cur_position_ids = [
                    torch.arange(
                        cur_inputs_embeds[-1].shape[0],
                        device=cur_inputs_embeds[-1].device,
                    )
                ]
                cur_labels = [labels[sorted_idx[i]][attention_mask[sorted_idx[i]]]]

        if len(cur_inputs_embeds):
            new_inputs_embeds.append(torch.cat(cur_inputs_embeds, 0))
            new_position_ids.append(torch.cat(cur_position_ids, 0))
            new_labels.append(torch.cat(cur_labels, 0))

        # print(new_position_ids[0].device, [x.shape for x in new_inputs_embeds], [x.shape for x in new_labels], [x.shape for x in new_position_ids])
        # assert 0
        new_inputs_embeds = torch.nn.utils.rnn.pad_sequence(
            new_inputs_embeds, batch_first=True, padding_value=self.get_llm().pad_token_id
        )

        new_position_ids = torch.nn.utils.rnn.pad_sequence(new_position_ids, batch_first=True, padding_value=-1)

        new_labels = torch.nn.utils.rnn.pad_sequence(new_labels, batch_first=True, padding_value=IGNORE_INDEX)
        ## yunhao: it's currently a workaround to avoid errors for seq_len < 100
        new_attention_mask = new_position_ids.ne(-1)
        # sanity check
        assert new_attention_mask.sum() == attention_mask.sum()
        # print(new_inputs_embeds.shape, (new_attention_mask.sum(1)))
        # print(sorted_seqlens_in_batch.device, sorted_seqlens_in_batch, new_attention_mask.sum(1))

        # return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
        return (
            None,
            new_position_ids,
            new_attention_mask,
            past_key_values,
            new_inputs_embeds,
            new_labels,
            sorted_seqlens_in_batch,
        )


    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.use_mm_patch_token:
            tokenizer.add_tokens(list(X_PATCH_TOKEN.values()), special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.use_mm_start_end:
            num_new_tokens = tokenizer.add_tokens(list(X_START_TOKEN.values())+list(X_END_TOKEN.values()), special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg
            ## TODO yunhao: handle cases for <im_st> <im_end>
            if model_args.pretrain_mm_connector:
                mm_connector_weights = torch.load(model_args.pretrain_mm_connector, map_location="cpu")
                embed_tokens_weight = mm_connector_weights["model.embed_tokens.weight"]
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.use_mm_patch_token:
            if model_args.mm_connector:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
