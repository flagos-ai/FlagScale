# sparse model change the parallel config

python convert.py \
    --model-type deepseek_v3 \
    --loader mcore \
    --saver transformers \
    --load-dir bf16_model \
    --save-dir converted_huggingface_model \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --target-expert-parallel-size 1 \
    --target-params-dtype bf16 \
    --true-vocab-size 151851 \

python convert.py \
    --model-type deepseek_v3 \
    --loader transformers \
    --saver mcore \
    --load-dir DeepSeek-V2-Lite  \
    --save-dir converted_bf16_model \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 2 \
    --target-decoder-first-pipeline-num-layers 13 \
    --target-expert-parallel-size 2 \
    --target-params-dtype bf16 \
    --true-vocab-size 151851 \

python convert.py \
    --model-type mixtral \
    --loader transformers \
    --saver mcore \
    --load-dir Mixtral-8x7B-v0.1 \
    --save-dir output \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 2 \
    --target-expert-parallel-size 2 \
    --target-params-dtype fp32 \
    --megatron-path <xxx>


# dense model convert to sparse model, mlp weight copy to all experts weight
# padding vocab_size with default value 64
python convert.py \
    --model-type mistral mixtral \
    --loader transformers \
    --saver mcore \
    --load-dir Mistral-7B-v0.1 \
    --save-dir output \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 2 \
    --target-expert-parallel-size 2 \
    --target-params-dtype fp32 \
    --target-num-experts 8 \
    --true-vocab-size 151851 \
    --megatron-path <xxx>

python convert.py \
    --model-type llama \
    --loader transformers \
    --saver mcore \
    --load-dir meta-llama3/Meta-Llama-3-8B \
    --save-dir outputs_llama3/checkpoint_mc \
    --target-tensor-parallel-size 4 \
    --target-pipeline-parallel-size 2 \
    --target-expert-parallel-size 1 \
    --target-params-dtype bf16 \
    --true-vocab-size 128256 \
    --megatron-path <xxx>

python convert.py \
    --model-type llama \
    --loader transformers \
    --saver mcore \
    --load-dir ${transformers_ckpt_path:??} \
    --save-dir ${mcore_ckpt_path:??} \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 4 \
    --target-expert-parallel-size 1 \
    --max-queue-size 50 \
    --target-params-dtype bf16 \
    --true-vocab-size 128256 \
    --megatron-path <xxx>

python convert.py \
    --model-type aquila3_dense \
    --loader transformers \
    --saver mcore \
    --load-dir $loaddir \
    --save-dir $outputs \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --target-expert-parallel-size 1 \
    --target-params-dtype bf16 \
    --true-vocab-size 151665 \
    --megatron-path <xxx>

# megatron to huggingface
python convert.py \
    --model-type llama \
    --loader mcore \
    --saver transformers \
    --load-dir $loaddir \
    --save-dir $outputs \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --target-expert-parallel-size 1 \
    --target-params-dtype bf16 \
    --true-vocab-size 128256 \
    --megatron-path <xxx>

python convert.py \
    --model-type qwen3 \
    --loader mcore \
    --saver transformers \
    --load-dir $loaddir \
    --save-dir $outputs \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --target-expert-parallel-size 1 \
    --target-params-dtype bf16 \
    --true-vocab-size 151851 \
    --megatron-path <xxx>
