export PYTHONPATH=$(dirname $(dirname "$0")):$(dirname $(dirname $(dirname "$0")))/flagscale/train/:$PYTHONPATH
# export MP_PP0_LAYERS=30
# bash hf2mcore_qwen3_vl_convertor.sh \
#     32B \
#     /share/project/lizhiyu/data/qwen3_vl/Qwen3-VL-32B-Instruct \
#     /share/project/lizhiyu/data/qwen3_vl/Qwen3-VL-32B-Instruct-tp4-pp2 \
#     4 \
#     2 \
#     false \
#     bf16 \
#     /share/project/lizhiyu/data/qwen3_vl/Qwen3-VL-32B-Instruct

bash hf2mcore_qwen3_vl_convertor.sh \
    8B \
    /share/project/lizhiyu/data/qwen3_vl/Qwen3-VL-8B-Instruct \
    /share/project/lizhiyu/data/qwen3_vl/Qwen3-VL-8B-Instruct-tp1 \
    1 \
    1 \
    false \
    bf16 \
    /share/project/lizhiyu/data/qwen3_vl/Qwen3-VL-8B-Instruct


# bash hf2mcore_qwen3_vl_convertor.sh \
#     4B \
#     /share/project/lizhiyu/data/qwen3_vl/Qwen3-VL-4B-Instruct-tp2 \
#     /share/project/lizhiyu/data/qwen3_vl/Qwen3-VL-4B-Instruct-tp2-hf \
#     2 \
#     1 \
#     true \
#     bf16 \
#     /share/project/lizhiyu/data/qwen3_vl/Qwen3-VL-4B-Instruct
