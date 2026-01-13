# Mechanism for Args Mapping

Generally, we map args from vllm style to a specific backend style (such as llama.cpp).

There are 2 types of mapping:
1. key mapping: only key changes
2. kv mapping: both key and value need changing

When converting:
1. flagscale/serve/args_mapping/mapping.yaml: configures the 2 types of mapping.
2. flagscale/serve/args_mapping/mapping_func/BACKEND_NAME.py: defines funcs needed when kv mapping.
3. flagscale/serve/args_mapping/mapping.py: entrypoint for converting.

# Add Mapping for a New Backend

## 1. Generate a conf template with all vLLM args.
```
cd FlagScale
python flagscale/serve/args_mapping/gen_template_funcs.py
```
output file: ./template_conf.yaml

## 2. Update the keys with "TODO" values in the template_conf.yaml file.
Possible choices are:
1. Delete, if no need for mapping
2. Update, if there is a key mapping
3. Move from "key_mapping" to "kv_mapping_func", if there is a kv mapping

## 3. Merge the updates to mapping.yaml.
Add a new key to mapping.yaml with backend name.

## 4. Generate a mapping_funcs template.

```
cd FlagScale
python flagscale/serve/args_mapping/gen_template_funcs.py --backend-name llama_cpp
```
output file: ./template_funcs.py

## 5. Implement the func in template_funcs.py.

template_funcs.py is generated after step 4.
