import os
import argparse
import logging
import traceback
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset, disable_caching

# Полностью отключить кэширование
disable_caching()

# Константы
CACHE_DIR = "path/to/cache_dir"
MODEL_ID = 'Qwen/Qwen3-8B-FP8'

def setup_environment():
    """Настройка переменных окружения"""
    os.environ['TORCH_HOME'] = CACHE_DIR
    os.environ["HF_HOME"] = CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
    os.environ["TORCH_HOME"] = CACHE_DIR
    

def translate_batch_to_ru(
    llm: LLM,
    texts: list[str],
    tokenizer,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    max_tokens: int = 1024,
    model_max_len: int = 2200,
    truncate_ratio: float = 0.95,
    verbose: bool = True,
):
    """
    Переводит батч текстов с английского на русский.
    """
    SYSTEM_PROMPT_TEMPLATE = """You are a professional technical translator.
Your task is translation text from English to Russian.

Important rules:
- DO NOT translate formulas, variable names, or code snippets.
- DO NOT translate identifiers inside backticks (`like_this`).
- DO NOT translate fenced code blocks (```python ... ```).
- Translate only the natural language description, comments, and documentation text.
- Keep formatting, indentation, and docstring style unchanged.

In the answer write ONLY translation of the following text.
Now translate the following text:
{text}
"""

    def build_and_truncate_prompt(text: str, idx: int) -> str:
        full_prompt = SYSTEM_PROMPT_TEMPLATE.replace("{text}", text)
        tokens = tokenizer.encode(full_prompt, add_special_tokens=False)
        token_count = len(tokens)

        limit = int((model_max_len - max_tokens) * truncate_ratio)

        if token_count > limit:
            if verbose:
                print(f" Text {idx + 1} truncated from {token_count} -> {limit} tokens (ratio={truncate_ratio:.2f}).")
            tokens = tokens[:limit]
            full_prompt = tokenizer.decode(tokens, skip_special_tokens=True)
        return full_prompt

    prompts = [build_and_truncate_prompt(t, i) for i, t in enumerate(texts)]
    messages_batch = [[{"role": "user", "content": p}] for p in prompts]

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )

    outputs = llm.chat(
        messages_batch,
        sampling_params,
        chat_template_kwargs={"enable_thinking": False},
    )

    translations = [out.outputs[0].text.strip() for out in outputs]
    return translations

def log_print(message: str, log_path: str):
    print(message)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def main():
    parser = argparse.ArgumentParser(description='Translate dataset')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for processing')
    parser.add_argument('--max_model_len', type=int, required=True, help='Maximum model context length')
    parser.add_argument('--max_tokens', type=int, required=True, help='Maximum tokens to generate')
    parser.add_argument('--enable_prefix_caching', type=bool, required=True, help='If use prefix caching. Useful when prompts have same prefixes')
    parser.add_argument('--gpu_memory_utilization', type=float, required=True, help='GPU memory utilization')
    parser.add_argument('--path_to_en_ds', type=str, required=True, help='Path to English dataset')
    parser.add_argument('--path_to_save', type=str, required=True, help='Path to save translated dataset')
    parser.add_argument('--text_field', type=str, required=True, help='Field of datasets.Dataset that we will translate')


    args = parser.parse_args()
    
    setup_environment()
    logging.basicConfig(level=logging.INFO)
    
    print(f"Initializing LLM with batch_size={args.batch_size}, max_model_len={args.max_model_len}, gpu_memory_utilization={args.gpu_memory_utilization}")
    llm = LLM(
        model=MODEL_ID, 
        gpu_memory_utilization=args.gpu_memory_utilization, 
        max_num_seqs=args.batch_size, 
        max_model_len=args.max_model_len, 
        enable_prefix_caching=args.enable_prefix_caching,
        download_dir=CACHE_DIR
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR)
    print("Tokenizer loaded from:", tokenizer.init_kwargs.get("_name_or_path"))
    
    # Создание директории для сохранения
    os.makedirs(args.path_to_save, exist_ok=True)
    
    # Функция перевода для map
    def translate(batch):
        # texts = batch['query']
        texts = batch[args.text_field]
        batch['ru_'+args.text_field] = translate_batch_to_ru(
            llm, 
            texts, 
            tokenizer,
            max_tokens=args.max_tokens,
            model_max_len=args.max_model_len
        )
        return batch
    
    # Логирование
    log_path = os.path.join(args.path_to_save, "translation_log.txt")
    print('log_path:', log_path)
    
    # Обработка шардов
    for shard_name in os.listdir(args.path_to_en_ds):
        # Пропускаем уже переведенные шарды
        if shard_name in os.listdir(args.path_to_save):
            log_print(f'Шард {shard_name} уже перевели, см. резы в папке', log_path)
            continue
        if shard_name.startswith('cornstack') and shard_name.endswith('.arrow'):
            log_print(f'Начинаем перевод {shard_name} шарда', log_path)

            try:
                # Загружаем шард
                shard = load_dataset(args.path_to_en_ds, data_files=[shard_name])

                shard = shard.map(
                    translate,
                    batched=True,
                    batch_size=args.batch_size,
                )
                shard.save_to_disk(os.path.join(args.path_to_save, shard_name))
                log_print(f'Шард {shard_name} успешно переведён и сохранён!', log_path)

            except Exception as e:
                error_message = f'Ошибка при обработке {shard_name}: {str(e)}\n{traceback.format_exc()}'
                log_print(error_message, log_path)

if __name__ == "__main__":
    main()