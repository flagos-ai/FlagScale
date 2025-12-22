## Running a Training Task

This tutorial demonstrates how to run a training task using Megatron-LM.

1. Prepare dataset demo:

   We provide a small processed data ([bin](https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.bin) and
   [idx](https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.idx))
   from the [Pile](https://pile.eleuther.ai/) dataset.

   ```shell
   mkdir -p /path/to/data && cd /path/to/data
   wget https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.idx
   wget https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.bin
   ```

1. Edit config:

   Modify the data path in `./examples/aquila/conf/train/7b.yaml`

   ```yaml
   data:
       data_path: ${data_path:??}  # modify data path here
       split: 1
       tokenizer:
           legacy_tokenizer: true
           tokenizer_type: AquilaTokenizerFS
           vocab_file: ./examples/aquila/tokenizer/vocab.json
           merge_file: ./examples/aquila/tokenizer/merges.txt
           special_tokens_file: ./examples/aquila/tokenizer/special_tokens.txt
           vocab_size: 100008
   ```

1. Start the distributed training job:

   ```shell
   python run.py --config-path ./examples/aquila/conf --config-name train action=run
   ```

1. Stop the distributed training job:

   ```shell
   python run.py --config-path ./examples/aquila/conf --config-name train action=stop
   ```
