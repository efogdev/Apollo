# This file is modified from https://github.com/haotian-liu/LLaVA/
from dataclasses import dataclass, field
from apollo.utils import mprint
import warnings
import os

from transformers import AutoConfig, PretrainedConfig, PreTrainedModel
from .hybrid_encoder import HybridVisionTower
from .siglip_encoder import SiglipVisionTower
from .intern_encoder import InternVisionTower
from .dino_encoder import DinoVisionTower
from .languagebind_encoder import LanguageBindImageTower, LanguageBindVideoTower
from .videomae_encoder import MAEVideoTower
from .internvideo_encoder import InternVideoTower

from .vjepa_encoder import VJepaVisionTower


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
    
    use_s2 = getattr(config, 's2', False)
    if "internimage" in encoder_name:
        if hasattr(config, 'drop_path_rate'):
            vision_tower = InternVisionTower(
                encoder_path, config=config, drop_path_rate=config.drop_path_rate)
        else:
            vision_tower = InternVisionTower(
                encoder_path, config=config, drop_path_rate=0.0)
    elif "radio" in encoder_name:
        vision_tower = RADIOVisionTower(encoder_path, config)
    elif "languagebind" in encoder_name:
        if "video" in encoder_name:
            #import ipdb; ipdb.set_trace()
            vision_tower = LanguageBindVideoTower(encoder_path, config, vision_tower_cfg)
        elif "image" in encoder_name:
            vision_tower = LanguageBindImageTower(encoder_path, config, vision_tower_cfg)
    elif "clip" in encoder_name:
        if use_s2:
            vision_tower = CLIPVisionTowerS2(encoder_path, config, vision_tower_cfg)
        else:
            vision_tower = CLIPVisionTower(encoder_path, config, vision_tower_cfg)
    elif "siglip" in encoder_name:
        if use_s2:
            vision_tower = SiglipVisionTowerS2(encoder_path, config, vision_tower_cfg)
        else:
            vision_tower = SiglipVisionTower(encoder_path, config, vision_tower_cfg)
    elif "dino" in encoder_name:
        vision_tower = DinoVisionTower(encoder_path, config, vision_tower_cfg)        
    elif "vjepa" in encoder_name:
        vision_tower = VJepaVisionTower(encoder_path, config, vision_tower_cfg)
    elif "ijepa" in encoder_name:
        vision_tower = IJepaVisionTower(encoder_path, config)
    elif "internvideo" in encoder_name:
        vision_tower = InternVideoTower(encoder_path, config, vision_tower_cfg)
    elif "videomae" in encoder_name:
        vision_tower = MAEVideoTower(encoder_path, config, vision_tower_cfg)
    else:
        raise ValueError(f"Unknown vision tower: {encoder_name}")

    return vision_tower


def build_vision_tower(
    model_name_or_path: str, config: PretrainedConfig
) -> PreTrainedModel:
    register_encoders()
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



def register_encoders():
    # Image
    clip_l_p16_384 = Encoder(
        encoder_name='clip-large-patch14-336',
        encoder_type='image',
        encoder_path='./model_zoo/CLIP/clip-vit-large-patch14-336',
    )
    add_encoder(clip_l_p16_384)

    siglip_so400m_p14_384 = Encoder(
        encoder_name='siglip-so400m-patch14-384',
        encoder_type='image',
        encoder_path='./model_zoo/SigLip/siglip-so400m-patch14-384',
    )
    add_encoder(siglip_so400m_p14_384)

    siglip_l_p16_384 = Encoder(
        encoder_name='siglip-large-patch16-384',
        encoder_type='image',
        encoder_path='./model_zoo/SigLip/siglip-large-patch16-384',
    )
    add_encoder(siglip_l_p16_384)

    siglip_b_p16_384 = Encoder(
        encoder_name='siglip-base-patch16-384',
        encoder_type='image',
        encoder_path='./model_zoo/SigLip/siglip-base-patch16-384',
    )
    add_encoder(siglip_b_p16_384)

    languagebind_image = Encoder(
        encoder_name='languagebind-image',
        encoder_type='image',
        encoder_path='./model_zoo/LanguageBind/LanguageBind_Image_Encoder',
        processor_path='./model_zoo/LanguageBind/LanguageBind_Image_Encoder',
    )
    add_encoder(languagebind_image)

    dinov2_l = Encoder(
        encoder_name='dinov2-l',
        encoder_type='image',
        encoder_path='./model_zoo/DINO/dinov2-large',
        processor_path='./model_zoo/DINO/dinov2-large',
    )
    add_encoder(dinov2_l)


    # Video

    vjepa_h_p16_384 = Encoder(
        encoder_name='vjepa_vith16-384',
        encoder_type='video',
        encoder_path='./model_zoo/VJEPA/jepa_vith16-384',
        processor_path='./model_zoo/VJEPA/jepa_vith16-384',
    )
    add_encoder(vjepa_h_p16_384)

    vjepa_h_p16_224 = Encoder(
        encoder_name='vjepa_vith16',
        encoder_type='video',
        encoder_path='./model_zoo/VJEPA/jepa_vith16.pth.tar',
        processor_path='./model_zoo/VJEPA/jepa_vith16',
    )
    add_encoder(vjepa_h_p16_224)

    vjepa_l_p16_224 = Encoder(
        encoder_name='vjepa_vitl16',
        encoder_type='video',
        encoder_path='./model_zoo/VJEPA/jepa_vitl16.pth.tar',
        processor_path='./model_zoo/VJEPA/jepa_vitl16',
    )
    add_encoder(vjepa_l_p16_224)

    languagebind_video = Encoder(
        encoder_name='languagebind-video-v1_5',
        encoder_type='video',
        encoder_path='./model_zoo/LanguageBind/LanguageBind_Video_Encoder',
        processor_path='./model_zoo/LanguageBind/LanguageBind_Video_Encoder',
    )
    add_encoder(languagebind_video)

    internvideo2 = Encoder(
        encoder_name='internvideo2',
        encoder_type='video',
        encoder_path='./model_zoo/InternVideo2-Stage2_1B-224p-f4',
        processor_path='./model_zoo/InternVideo2-Stage2_1B-224p-f4',
    )
    add_encoder(internvideo2)

    videomae_l = Encoder(
        encoder_name='videomae-l',
        encoder_type='video',
        encoder_path='./model_zoo/VideoMAE/videomae-large',
        processor_path='./model_zoo/VideoMAE/videomae-large',
    )
    add_encoder(videomae_l)

    