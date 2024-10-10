# 需要huggingface-cli login后才行
export HF_ENDPOINT=https://hf-mirror.com
base_dir="/mnt/bn/lys-lq/HF_Caches"
model_name_or_path="bert-large-uncased"
mkdir -p "${base_dir}/${model_name_or_path}"
huggingface-cli download --local-dir-use-symlinks False "${model_name_or_path}" --local-dir "${base_dir}/${model_name_or_path}"      