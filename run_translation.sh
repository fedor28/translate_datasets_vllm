# Скрипт для запуска перевода датасета

BATCH_SIZE=128
MAX_MODEL_LEN=2200
MAX_TOKENS=1024
ENABLE_PREFIX_CACHING=True
GPU_MEMORY_UTILIZATION=0.9
PATH_TO_EN_DS="path/to/source_dataset"
PATH_TO_SAVE="path/to/save/translated_dataset"
TEXT_FIELD="query"

CUDA_VISIBLE_DEVICES=1 python vllm_translate_dataset.py \
    --batch_size $BATCH_SIZE \
    --max_model_len $MAX_MODEL_LEN \
    --max_tokens $MAX_TOKENS \
    --enable_prefix_caching $ENABLE_PREFIX_CACHING\
    --gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
    --path_to_en_ds $PATH_TO_EN_DS \
    --path_to_save $PATH_TO_SAVE \
    --text_field $TEXT_FIELD