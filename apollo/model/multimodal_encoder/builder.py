# This file is modified from https://github.com/haotian-liu/LLaVA/
from dataclasses import dataclass, field
from apollo.utils import mprint
import warnings
import os

from transformers import AutoConfig, PretrainedConfig, PreTrainedModel
from .hybrid_encoder import HybridVisionTower
from .siglip_encoder import SiglipVisionTower
from .internvideo_encoder import InternVideoTower


def build_single_vision_tower(encoder_name, encoder_path, config: PretrainedConfig
    ) -> PreTrainedModel:
    ## skip vision tower instantiation
    #import ipdb; ipdb.set_trace()
    vision_tower_arch = None
    if config.resume_path:
        encoder_path = os.path.join(encoder_path, encoder_name)
        assert os.path.exists(
            encoder_path
        ), f"Resume vision tower path {encoder_path} does not exist!"
        vision_tower_cfg = AutoConfig.from_pretrained(encoder_path, trust_remote_code=True)
        vision_tower_arch = vision_tower_cfg.architectures[0].lower()
    else:
        vision_tower_cfg = None
    vision_tower_name = (
        vision_tower_arch if vision_tower_arch is not None else encoder_path
    )
    
    if "siglip" in encoder_name:
        vision_tower = SiglipVisionTower(encoder_path, config, vision_tower_cfg)
    elif "internvideo" in encoder_name:
        vision_tower = InternVideoTower(encoder_path, config, vision_tower_cfg)
    else:
        raise ValueError(f"Unknown vision tower: {encoder_name}")

    return vision_tower


def build_vision_tower(
    model_name_or_path: str, config: PretrainedConfig
) -> PreTrainedModel:
    ## skip vision tower instantiation
    if config.resume_path:
        hybrid_tower_cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        model_names_or_path = hybrid_tower_cfg.vision_towers
        mprint(f"[Encoders-INFO]: Using from {model_names_or_path}")
        vision_towers = {}
        configs = {}
        encoders = [Encoder(encoder_name, encoder_path=model_name_or_path) for encoder_name in model_names_or_path]

    else:
        model_names_or_path = model_name_or_path.strip().split('+')
        encoders = (ENCODERS[_] for _ in model_names_or_path)

    mprint(f"[Encoders-INFO]: Using from {model_names_or_path}")
    
    vision_towers = {}
    configs = {}
    for encoder in encoders:
        vision_tower = build_single_vision_tower(encoder.encoder_name, encoder.encoder_path, config)
        vision_towers[encoder.encoder_name] = vision_tower
        configs[encoder.encoder_name] = vision_tower.config

    return HybridVisionTower(vision_towers, configs)



@dataclass
class Encoder:
    encoder_name: str
    encoder_type: str = field(default="image")
    encoder_path: str = field(default=None, metadata={"help": "Path to the encoder"})
    processor_path: str = field(default=None, metadata={"help": "Path to the processor"})
    meta_path: str = field(default=None, metadata={"help": "Path to the meta data for webdataset."})
    description: str = field(
        default=None,
        metadata={
            "help": "Detailed desciption of where the data is from, how it is labelled, intended use case and the size of the dataset."
        },
    )



ENCODERS = {}



def add_encoder(encoder):
    if encoder.encoder_name in ENCODERS:
        # make sure the data_name is unique
        warnings.warn(f"{encoder.encoder_name} already existed in DATASETS. Make sure the name is unique.")
    assert "+" not in encoder.encoder_name, "Dataset name cannot include symbol '+'."
    ENCODERS.update({encoder.encoder_name: encoder})

