export PYTHONPATH=$PYTHONPATH:../../../:../../../flagscale/train/

python convert_custom_dataset_to_wds_chatml_str.py \
    --dataset-root=/share/project/lizhiyu/LLaMA-Factory/data/sample_dataset/text_only/ \
    --output-root=/share/project/lizhiyu/LLaMA-Factory/data/sample_dataset/text_only/ \
    --json=text_only_samples_10_first.json \
    --train-split 1 \
    --val-split 0 \
    --images-key=image \
    --videos-key=video \
    --vision-root=/mnt/LLaVA-Pretrain \
    --max-samples-per-tar 100000000 \
    --dp-size 1 \
    --num-workers 20
