import os, re, sys
import spaces
import traceback
import gradio as gr

import torch
import numpy as np
from num2words import num2words
from datetime import timedelta
import datetime


from apollo.builder import load_pretrained_model
from apollo.constants import (
    X_TOKEN,
    X_TOKEN_INDEX,
)
from apollo.conversation import conv_templates, SeparatorStyle
from apollo.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_mm_token,
    ApolloMMLoader
)
from decord import cpu, VideoReader
from huggingface_hub import snapshot_download


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Ensure the environment has GPU support.")
else:
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")


token = os.getenv("HUGGINGFACE_API_KEY")


title_markdown = """
<div style="display: flex; justify-content: center; align-items: center; text-align: center;">
  <div>
    <h1 >You are chatting with Apollo-3B</h1>
  </div>
</div>
<div align="center">
    <div style="display:flex; gap: 0.25rem; margin-top: 10px;" align="center">
        <a href='https://apollo-lmms.github.io/Apollo/'><img src='https://img.shields.io/badge/Project-Apollo-deepskyblue'></a>
        <a href='https://huggingface.co/Apollo-LMMs/Apollo-3B'><img src='https://img.shields.io/badge/model-checkpoints-gold'></a>
    </div>
</div>
"""

block_css = """
#buttons button {
    min-width: min(120px,100%);
    color: #9C276A
}
"""

plum_color = gr.themes.colors.Color(
    name='plum',
    c50='#F8E4EF',
    c100='#E9D0DE',
    c200='#DABCCD',
    c300='#CBA8BC',
    c400='#BC94AB',
    c500='#AD809A',
    c600='#9E6C89',
    c700='#8F5878',
    c800='#804467',
    c900='#713056',
    c950='#662647',
)


model_path = snapshot_download("Apollo-LMMs/Apollo-3B-chatty", repo_type="model", use_auth_token=token)
data_path = snapshot_download("Apollo-LMMs/examples", repo_type="dataset", use_auth_token=token)

class Chat:
    def __init__(self):
        self.version = "qwen_1_5"
        model_name = "apollo"
        
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = get_model_name_from_path(model_path)
        attn_implementation="sdpa" if torch.__version__ > "2.1.2" else "eager"
        
        self._tokenizer, self._model, self._vision_processors, self._max_length = load_pretrained_model(
            model_path, 
            model_name, 
            device=device, 
            attn_implementation=attn_implementation)
        
        self._model = self._model.to(torch.bfloat16)
        self._config = self._model.config
        self.num_repeat_token = self._config.mm_connector_cfg.num_output_tokens #todo: get from config
        self.mm_use_im_start_end = self._config.use_mm_start_end
        self.vision_processors = self._vision_processors
        
        num_frames = []
        for vision_tower in self._model.get_vision_tower().vision_towers:
            vision_tower = getattr(self._model.get_vision_tower(), vision_tower)
            num_frames.append(getattr(vision_tower, 'num_frames', 1))
        
        frames_per_clip = max(num_frames)
        clip_duration=getattr(self._config, 'clip_duration')

        self.mm_processor =  ApolloMMLoader(self._vision_processors, 
                                            clip_duration, 
                                            frames_per_clip, 
                                            clip_sampling_ratio=0.35,
                                            device=device,
                                            num_repeat_token=self.num_repeat_token)
        
        self._model.config.encode_batch_size = 10
        self._model.eval()

    def remove_after_last_dot(self, s):
        last_dot_index = s.rfind('.')
        if last_dot_index == -1:
            return s
        return s[:last_dot_index + 1]

    def apply_first_prompt(self, message, replace_string, data_type):
        if self.mm_use_im_start_end:
            message = X_START_TOKEN[data_type] + replace_string + X_END_TOKEN[data_type] + '\n\n' + message
        else:
            message = (replace_string) + '\n\n' + message

        return message
    
    @spaces.GPU(duration=120)
    @torch.inference_mode()
    def generate(self, data: list, message, temperature, top_p, max_output_tokens):
        # TODO: support multiple turns of conversation.
        mm_data, replace_string, data_type = data[0]
        print(message)
        
        conv = conv_templates[self.version].copy() 
        if isinstance(message, str):
            message = self.apply_first_prompt(message, replace_string, data_type)
            conv.append_message(conv.roles[0], message)
        elif isinstance(message, list):
            if X_TOKEN[data_type] not in message[0]['content']:
                print('applying prompt')
                message[0]['content'] = self.apply_first_prompt(message[0]['content'], replace_string, data_type)
            
            for mes in message:
                conv.append_message(mes["role"], mes["content"])
                
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        print(prompt.replace(X_TOKEN['video'],'<v>'))
        input_ids = tokenizer_mm_token(prompt, self._tokenizer, return_tensors="pt").unsqueeze(0).cuda().to(self._model.device)
        
        pad_token_ids = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else self._tokenizer.eos_token_id
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self._tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self._model.generate(input_ids,
                                            vision_input=[mm_data], 
                                            data_types=[data_type], 
                                            do_sample=True if temperature > 0 else False,
                                            temperature=temperature,
                                            max_new_tokens=max_output_tokens, 
                                            top_p=top_p,
                                            use_cache=True, 
                                            num_beams=1,
                                            stopping_criteria=[stopping_criteria])
        
        pred = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return self.remove_after_last_dot(pred)


@spaces.GPU(duration=120)
def generate(image, video, message, chatbot, textbox_in, temperature, top_p, max_output_tokens, dtype=torch.float16):
    print(message)
    if textbox_in is None:
        raise gr.Error("Chat messages cannot be empty")
        return (
            gr.update(value=image, interactive=True),
            gr.update(value=video, interactive=True),
            message,
            chatbot,
            None,
        )
    data = []

    mm_processor = handler.mm_processor
    try:
        if image is not None:
            image, prompt = mm_processor.load_image(image)
            data.append((image, prompt, 'image'))
        elif video is not None:
            video_tensor, prompt = mm_processor.load_video(video)
            data.append((video_tensor, prompt, 'video'))
            
        elif image is None and video is None:
            data.append((None, None, 'text'))
        else:
            raise NotImplementedError("Not support image and video at the same time")
            
    except Exception as e:
        traceback.print_exc()
        return gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), message, chatbot, None

    assert len(message) % 2 == 0, "The message should be a pair of user and system message."

    show_images = ""
    if image is not None:
        show_images += f'<img src="./file={image}" style="display: inline-block;width: 250px;max-height: 400px;">'
    if video is not None:
        show_images += f'<video controls playsinline width="300" style="display: inline-block;"  src="./file={video}"></video>'

    one_turn_chat = [textbox_in, None]

    # 1. first run case
    if len(chatbot) == 0:
        one_turn_chat[0] += "\n" + show_images
    # 2. not first run case
    else:
        # scanning the last image or video
        length = len(chatbot)
        for i in range(length - 1, -1, -1):
            previous_image = re.findall(r'<img src="./file=(.+?)"', chatbot[i][0])
            previous_video = re.findall(r'<video controls playsinline width="500" style="display: inline-block;"  src="./file=(.+?)"', chatbot[i][0])

            if len(previous_image) > 0:
                previous_image = previous_image[-1]
                # 2.1 new image append or pure text input will start a new conversation
                if (video is not None) or (image is not None and os.path.basename(previous_image) != os.path.basename(image)):
                    message.clear()
                    one_turn_chat[0] += "\n" + show_images
                break
            elif len(previous_video) > 0:
                previous_video = previous_video[-1]
                # 2.2 new video append or pure text input will start a new conversation
                if image is not None or (video is not None and os.path.basename(previous_video) != os.path.basename(video)):
                    message.clear()
                    one_turn_chat[0] += "\n" + show_images
                break

    message.append({'role': 'user', 'content': textbox_in})
    text_en_out = handler.generate(data, message, temperature=temperature, top_p=top_p, max_output_tokens=max_output_tokens)
    message.append({'role': 'assistant', 'content': text_en_out})

    one_turn_chat[1] = text_en_out
    chatbot.append(one_turn_chat)

    return gr.update(value=image, interactive=True), gr.update(value=video, interactive=True), message, chatbot, None


def regenerate(message, chatbot):
    message.pop(-1), message.pop(-1)
    chatbot.pop(-1)
    return message, chatbot


def clear_history(message, chatbot):
    message.clear(), chatbot.clear()
    return (gr.update(value=None, interactive=True),
            gr.update(value=None, interactive=True),
            message, chatbot,
            gr.update(value=None, interactive=True))

handler = Chat()

textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)

theme = gr.themes.Default(primary_hue=plum_color)
# theme.update_color("primary", plum_color.c500)
theme.set(slider_color="#9C276A")
theme.set(block_title_text_color="#9C276A")
theme.set(block_label_text_color="#9C276A")
theme.set(button_primary_text_color="#9C276A")

with gr.Blocks(title='Apollo-3B', theme=theme, css=block_css) as demo:
    gr.Markdown(title_markdown)
    message = gr.State([])

    with gr.Row():
        with gr.Column(scale=3):
            image = gr.State(None)
            video = gr.Video(label="Input Video")

            with gr.Accordion("Parameters", open=True) as parameter_row:

                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.4,
                    step=0.1,
                    interactive=True,
                    label="Temperature",
                )

                top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        interactive=True,
                        label="Top P",
                )

                max_output_tokens = gr.Slider(
                    minimum=64,
                    maximum=1024,
                    value=256,
                    step=64,
                    interactive=True,
                    label="Max output tokens",
                )

        with gr.Column(scale=7):
            chatbot = gr.Chatbot(label="Apollo", bubble_full_width=True, height=420)
            with gr.Row():
                with gr.Column(scale=8):
                    textbox.render()
                with gr.Column(scale=1, min_width=50):
                    submit_btn = gr.Button(value="Send", variant="primary", interactive=True)
            with gr.Row(elem_id="buttons") as button_row:
                upvote_btn     = gr.Button(value="👍  Upvote", interactive=True)
                downvote_btn   = gr.Button(value="👎  Downvote", interactive=True)
                regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=True)
                clear_btn      = gr.Button(value="🗑️  Clear history", interactive=True)

    with gr.Row():
        with gr.Column():
            gr.Examples(
                examples=[
                    [
                        f"./{data_path}/example1.mp4",
                        "At what time in the video is Peter Thompson interviewed? Respond in seconds, and describe what he is wearing.",
                    ],
                    [
                        f"./{data_path}/example2.mp4",
                        "What watch brands appear in the video?",
                    ],
                    [
                        f"./{data_path}/example3.mp4",
                        "What are the two people discussing?",
                    ],
                ],
                inputs=[video, textbox],
            )

    submit_btn.click(
        generate, 
        [image, video, message, chatbot, textbox, temperature, top_p, max_output_tokens],
        [image, video, message, chatbot, textbox])

    textbox.submit(
        generate,
        [
            image,
            video,
            message,
            chatbot,
            textbox,
            temperature,
            top_p,
            max_output_tokens,
        ],
        [image, video, message, chatbot, textbox],
    )

    regenerate_btn.click(
        regenerate, 
        [message, chatbot], 
        [message, chatbot]).then(
        generate, 
        [image, video, message, chatbot, textbox, temperature, top_p, max_output_tokens], 
        [image, video, message, chatbot])

    clear_btn.click(
        clear_history, 
        [message, chatbot],
        [image, video, message, chatbot, textbox])

demo.launch()
