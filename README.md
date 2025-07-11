## 1.5B train
```sh
export WANDB_API_KEY=""
export WANDB_PROJECT=""
export HF_TOKEN=""
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES="0,1,2,3"
ACCELERATE_LOG_LEVEL=info accelerate launch \
--main_process_port 12389 \
  --config_file recipes/accelerate_configs/zero2.yaml \
  --num_processes=3  \
  src/open_r1/grpo.py \
  --config recipes/gaussian_1.5b.yaml
```

## 1.5B evaluation
```sh
MODEL=""
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=""

lighteval vllm "$MODEL_ARGS" \
    "custom|aime24|0|0,custom|math_500|0|0,custom|amc23|0|0,custom|minerva|0|0,custom|olympiadbench|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir "$OUTPUT_DIR"
```

## 7B evaluation
```sh
MODEL=
MODEL_NAME=$(basename "$MODEL")
export CUDA_VISIBLE_DEVICES="1"
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=4096,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
lighteval vllm "$MODEL_ARGS" \
  "custom|aime24|0|0,custom|amc23|0|0,custom|math_500|0|0,custom|minerva|0|0,custom|olympiadbench|0|0" \
  --custom-tasks src/open_r1/evaluate_7b.py \
  --use-chat-template \
  --output-dir "$OUTPUT_DIR"
```
