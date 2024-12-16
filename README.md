```shell
pip -r requirements.txt
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install flash-attn --no-build-isolation
python app.py
```
