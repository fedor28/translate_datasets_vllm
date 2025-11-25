# translate_datasets_vllm
This repo contains the code for simple translation of HuggingFace datasets using Qwen-3 8B using [vllm](https://docs.vllm.ai/en/latest/).
You can easily use this code for translation tasks or other tasks using this LLM inference.

### How to
Clone repo 
```bash
git clone git@github.com:fedor28/translate_datasets_vllm.git
cd translate_datasets_vllm
```
  
Create venv  
```bash
python -m venv venv
source venv/bin/activate
```
  
Install requirements
```bash
pip install -r requirements.txt
```

### Detailed inference
Changes arguments of inference and run
```bash
./run_translation.sh
```
