#    Copyright 2024 Hao Zhang
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
from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn

from transformers import AutoConfig, PretrainedConfig, PreTrainedModel, AutoModelForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from apollo.model.apollo_arch import ApolloMetaModel, ApolloMetaForCausalLM
from ..configuration_apollo import ApolloConfig


class ApolloModel(ApolloMetaModel, PreTrainedModel):
    config_class = ApolloConfig 
    main_input_name = "input_embeds"
    supports_gradient_checkpointing = True

    def __init__(self, config: ApolloConfig = None, *args, **kwargs) -> None:
        super().__init__(config)
        return self.init_vlm(config=config, *args, **kwargs)


class ApolloForCausalLM(ApolloMetaForCausalLM, PreTrainedModel):
    config_class = ApolloConfig 
    supports_gradient_checkpointing = True

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)

        config.model_type = "apollo"
        config.rope_scaling = None

        self.model = ApolloModel(config, *args, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        vision_input: Optional[List[torch.FloatTensor]] = None,
        data_types: Optional[List[str]] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        self.freezed_module_patch()
        if inputs_embeds is None:
            (
                input_ids, 
                position_ids, 
                attention_mask, 
                past_key_values, 
                inputs_embeds, 
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, 
                position_ids, 
                attention_mask, 
                past_key_values, 
                labels, 
                vision_input, 
                data_types
            )

        return self.get_llm().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        vision_input: Optional[List[torch.Tensor]] = None,
        data_types: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if vision_input is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, vision_input, data_types=data_types)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return self.get_llm().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        vision_input = kwargs.pop("vision_input", None)
        data_types = kwargs.pop("data_types", None)
        inputs = self.get_llm().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if vision_input is not None:
            inputs["vision_input"] = vision_input
        if data_types is not None:
            inputs["data_types"] = data_types
        return inputs
        
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        if hasattr(cls, "load_pretrained"):
            return cls.load_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs,
            )
        return super(ApolloQwenForCausalLM).from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )


AutoConfig.register("apollo", ApolloConfig)
AutoModelForCausalLM.register(ApolloConfig, ApolloForCausalLM)