export PYTHONPATH=$PYTHONPATH:../../../:../../../third_party/Megatron-LM/
bash hf2mcore_qwen2.5_vl_convertor.sh 7B \
/share/project/lizhiyu/data/Qwen2.5-VL-7B-Instruct \
/share/project/lizhiyu/data/Qwen2.5-VL-7B-Instruct-tp2 \
2 1 false bf16  \
/share/project/lizhiyu/data/Qwen2.5-VL-7B-Instruct
