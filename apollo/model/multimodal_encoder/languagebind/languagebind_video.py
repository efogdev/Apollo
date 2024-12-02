import copy, os
from typing import Optional, Tuple, Union

import torch
from einops import rearrange
from torch import nn
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPMLP, CLIPAttention
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings, logging
from transformers import PretrainedConfig, PreTrainedModel
#from transformers.modeling_utils import 

logger = logging.get_logger(__name__)


class LanguageBindVideoEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CLIPVisionModel`]. It is used to instantiate a
    CLIP vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the CLIP
    [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from transformers import CLIPVisionConfig, CLIPVisionModel

    >>> # Initializing a CLIPVisionConfig with openai/clip-vit-base-patch32 style configuration
    >>> configuration = CLIPVisionConfig()

    >>> # Initializing a CLIPVisionModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
    >>> model = CLIPVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "clip_vision_model"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        projection_dim=512,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=32,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,

        add_time_attn=False, ################################
        num_frames=8, ################################
        force_patch_dropout=0.0, ################################
        lora_r=2, ################################
        lora_alpha=16, ################################
        lora_dropout=0.0, ################################
        num_mel_bins=0.0, ################################
        target_length=0.0, ################################
        video_decode_backend='decord', #########################
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

        self.add_time_attn = add_time_attn  ################
        self.num_frames = num_frames  ################
        self.force_patch_dropout = force_patch_dropout  ################
        self.lora_r = lora_r  ################
        self.lora_alpha = lora_alpha  ################
        self.lora_dropout = lora_dropout  ################
        self.num_mel_bins = num_mel_bins  ################
        self.target_length = target_length  ################
        self.video_decode_backend = video_decode_backend  ################

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from CLIPConfig
        if config_dict.get("model_type") == "clip":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)

class LanguageBindVideoConfig(PretrainedConfig):
    r"""
    [`CLIPConfig`] is the configuration class to store the configuration of a [`CLIPModel`]. It is used to instantiate
    a CLIP model according to the specified arguments, defining the text model and vision model configs. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the CLIP
    [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`CLIPTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`CLIPVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original CLIP implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import CLIPConfig, CLIPModel

    >>> # Initializing a CLIPConfig with openai/clip-vit-base-patch32 style configuration
    >>> configuration = CLIPConfig()

    >>> # Initializing a CLIPModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
    >>> model = CLIPModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a CLIPConfig from a CLIPTextConfig and a CLIPVisionConfig
    >>> from transformers import CLIPTextConfig, CLIPVisionConfig

    >>> # Initializing a CLIPText and CLIPVision configuration
    >>> config_text = CLIPTextConfig()
    >>> config_vision = CLIPVisionConfig()

    >>> config = CLIPConfig.from_text_vision_configs(config_text, config_vision)
    ```"""

    model_type = "LanguageBindVideoEncoder"
    is_composition = True

    def __init__(
        self, vision_config=None, projection_dim=512, logit_scale_init_value=2.6592, **kwargs
    ):
        # If `_config_dict` exist, we use them for the backward compatibility.
        # We pop out these 2 attributes before calling `super().__init__` to avoid them being saved (which causes a lot
        # of confusion!).
        vision_config_dict = kwargs.pop("vision_config_dict", None)

        super().__init__(**kwargs)
        if vision_config_dict is not None:
            if vision_config is None:
                vision_config = {}

            # This is the complete result when using `vision_config_dict`.
            _vision_config_dict = LanguageBindVideoEncoderConfig(**vision_config_dict).to_dict()
            # convert keys to string instead of integer
            if "id2label" in _vision_config_dict:
                _vision_config_dict["id2label"] = {
                    str(key): value for key, value in _vision_config_dict["id2label"].items()
                }

            # Give a warning if the values exist in both `_vision_config_dict` and `vision_config` but being different.
            for key, value in _vision_config_dict.items():
                if key in vision_config and value != vision_config[key] and key not in ["transformers_version"]:
                    # If specified in `vision_config_dict`
                    if key in vision_config_dict:
                        message = (
                            f"`{key}` is found in both `vision_config_dict` and `vision_config` but with different "
                            f'values. The value `vision_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`vision_config_dict` is provided which will be used to initialize `CLIPVisionConfig`. "
                            f'The value `vision_config["{key}"]` will be overriden.'
                        )
                    logger.warning(message)

            # Update all values in `vision_config` with the ones in `_vision_config_dict`.
            vision_config.update(_vision_config_dict)


        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. initializing the `CLIPVisionConfig` with default values.")

        self.vision_config = LanguageBindVideoEncoderConfig(**vision_config)

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: LanguageBindVideoEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # (b t) c h w
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)  # b hw c
        return embeddings

class CLIPVisionEmbeddings3D(nn.Module):
    def __init__(self, config: LanguageBindVideoEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_frames = config.num_frames
        self.tube_size = getattr(config, 'tube_size', 1)

        class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        self.class_embedding = nn.Parameter(class_embedding.data.repeat(self.num_frames // self.tube_size, 1))
        
        self.patch_embedding = nn.Conv3d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=(self.tube_size, self.patch_size, self.patch_size),
            stride=(self.tube_size, self.patch_size, self.patch_size),
            bias=False, 
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

        #self.expand3d()

    def expand3d(self):
        import ipdb; ipdb.set_trace()
        state_dict = self.patch_embedding.state_dict()
        print(state_dict['weight'].shape) #torch.Size([1024, 3, 14, 14])
        state_dict_expand = state_dict['weight'].unsqueeze(2)
        print(state_dict_expand.shape) #torch.Size([1024, 3, 1, 14, 14])
        device, dtype = state_dict_expand.device, state_dict_expand.dtype
        # print(device, dtype)

        zero = torch.zeros_like(state_dict_expand).to(device=device, dtype=dtype)
        state_dict_expand3d = torch.cat([state_dict_expand] + (self.tube_size-1)*[zero], dim=2)
        print(state_dict_expand3d.shape) #torch.Size([1024, 3, 1, 14, 14])

        patch_embedding = nn.Conv3d(
            in_channels=self.patch_embedding.in_channels,
            out_channels=self.embed_dim,
            kernel_size=(self.tube_size, self.patch_size, self.patch_size),
            stride=(self.tube_size, self.patch_size, self.patch_size),
            bias=False,
        ).to(device=device, dtype=dtype)
        patch_embedding.load_state_dict({'weight': state_dict_expand3d})
        class_embedding = nn.Parameter(self.class_embedding.data.repeat(self.num_frames // self.tube_size, 1)).to(device=device, dtype=dtype)
        
        self.patch_embedding = patch_embedding
        self.class_embedding = class_embedding


    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # (b t) c h w
        batch_size = pixel_values.shape[0] // self.num_frames
        pixel_values = rearrange(pixel_values, '(b t) c h w -> b c t h w', b=batch_size, t=self.num_frames)
        # print('pixel_values', pixel_values.shape)
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, t, grid, grid]
        # print('patch_embeds', patch_embeds.shape)
        # SET_GLOBAL_VALUE('NUM_FRAMES', patch_embeds.shape[2])
        patch_embeds = rearrange(patch_embeds, 'b c t h w -> b t (h w) c')

        class_embeds = self.class_embedding.unsqueeze(1).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # b t 1 c
        # print('class_embeds', class_embeds.device, class_embeds.dtype)
        # print('patch_embeds', patch_embeds.device, patch_embeds.dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=2)  # b t hw+1 c
        embeddings = embeddings + self.position_embedding(self.position_ids)
        embeddings = rearrange(embeddings, 'b t hw_1 c -> (b t) hw_1 c')
        return embeddings

class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x, B, T):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        if T == 1:
            rand = torch.randn(batch, num_tokens)
            patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices
        else:
            rand = torch.randn(B, num_tokens)
            patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices
            patch_indices_keep = patch_indices_keep.unsqueeze(1).repeat(1, T, 1)
            patch_indices_keep = rearrange(patch_indices_keep, 'b t n -> (b t) n')


        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x

class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: LanguageBindVideoConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.add_time_attn = config.add_time_attn
        if self.add_time_attn:
            self.t = config.num_frames
            self.temporal_embedding = nn.Parameter(torch.zeros(1, config.num_frames, config.hidden_size))
            nn.init.normal_(self.temporal_embedding, std=config.hidden_size ** -0.5)

            self.embed_dim = config.hidden_size
            self.temporal_attn = CLIPAttention(config)
            self.temporal_layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
            # self.temporal_mlp = CLIPMLP(config)
            # self.temporal_layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """


        if self.add_time_attn:
            bt, n, d = hidden_states.shape
            t = self.t

            # time embed
            if t != 1:
                n = hidden_states.shape[1]
                hidden_states = rearrange(hidden_states, '(b t) n d -> (b n) t d', t=t)
                hidden_states = hidden_states + self.temporal_embedding[:, :t, :]
                hidden_states = rearrange(hidden_states, '(b n) t d -> (b t) n d', n=n)

            # time attn
            residual = hidden_states
            hidden_states = rearrange(hidden_states, '(b t) n d -> (b n) t d', t=t)
            # hidden_states = self.layer_norm1(hidden_states)  # share layernorm
            hidden_states = self.temporal_layer_norm1(hidden_states)
            hidden_states, attn_weights = self.temporal_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = residual + rearrange(hidden_states, '(b n) t d -> (b t) n d', n=n)

            # residual = hidden_states
            # hidden_states = rearrange(hidden_states, '(b t) n d -> (b n) t d', t=t)
            # # hidden_states = self.layer_norm2(hidden_states)  # share layernorm
            # hidden_states = self.temporal_layer_norm2(hidden_states)
            # hidden_states = self.temporal_mlp(hidden_states)
            # hidden_states = residual + rearrange(hidden_states, '(b n) t d -> (b t) n d', n=n)

        # spatial attn
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


CLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

class CLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """

    def __init__(self, config: LanguageBindVideoConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class LanguageBindVideoEncoder(PreTrainedModel):
    config_class = LanguageBindVideoEncoderConfig
    def __init__(self, config: LanguageBindVideoEncoderConfig):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size
        vl_new = getattr(config, 'clip_type', 'vl') == 'vl_new'
        add_time_attn = config.add_time_attn
        # self.embeddings = CLIPVisionEmbeddings(config)
        if add_time_attn:
            if vl_new:
                self.embeddings = CLIPVisionEmbeddings3D(config)
            else:
                self.embeddings = CLIPVisionEmbeddings(config)

        self.patch_dropout = PatchDropout(config.force_patch_dropout)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=LanguageBindVideoEncoderConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        ######################################
        if len(pixel_values.shape) == 7:
            b_new, pair_new, T, bs_new, channel_new, h_new, w_new = pixel_values.shape
            # print(pixel_values.shape)
            B = b_new * pair_new * bs_new
            pixel_values = pixel_values.reshape(B*T, channel_new, h_new, w_new)

        elif len(pixel_values.shape) == 5:
            B, _, T, _, _ = pixel_values.shape
            # print(pixel_values.shape)
            pixel_values = rearrange(pixel_values, 'b c t h w -> (b t) c h w')
        else:
            # print(pixel_values.shape)
            B, _, _, _ = pixel_values.shape
            T = 1
        ###########################
        hidden_states = self.embeddings(pixel_values)

        hidden_states = self.patch_dropout(hidden_states, B, T)  ##############################################

        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        pooled_output = pooled_output.reshape(B, T, -1).mean(1) ################################
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
