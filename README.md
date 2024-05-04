# NLLB with CTranslate2
This is an example repository to convert Transformers-based NLLB models and run inference with CTranslate2 runtime. CTranslate2 is a project to apply performance optimization to accelerate and reduce the memory usage of Transformer models on CPU and GPU. Learn more about the project via their [official repository](https://github.com/OpenNMT/CTranslate2).

## Requirements
For this to work, you need these 2 major dependencies:
```txt
transformers>=4.21.0
ctranslate2
```
Though, if you don't wish to create a python virtual environment, you could also opt to build the docker image via the provided `Dockerfile`. Simply run:
```sh
docker build -t dleongsh/ctranslate-nllb:0.0.1 .
```
Then start up the docker container via the `docker-compose.yaml` file and enter the interactive bash terminal within.
```sh
docker-compose run --rm nllb bash
```

## Conversion of Model Weights
This repository assumes that you already have the NLLB model weights and files (in HuggingFace Transformers format). Update the source and target model directories in `src/utils/convert.sh`, and then execute it.
```sh
bash ./convert.sh
```
This repository also assumes you have the tokenizer files downloaded as well. Put them in the tokenizer folder. Your final input directory should look like this:
```
|- model_dir
|   |- model
|   |   |- config.json
|   |   |- model.bin
|   |   |- shared_vocabulary.json
|   |- tokenizer
|   |   |- sentencepiece.bpe.model
|   |   |- special_tokens_map.json
|   |   |- tokenizer_config.json
|   |   |- tokenizer.json
```

## Test Inference
In `src/main.py`, switch up these variables for your own.
```
MODEL_PATH = "/pretrained_models/nllb-200-distilled-600m-lora-ct2-f16-zh"
TEXT = "我很可爱，你知道吗？"
SOURCE_LANGUAGE = "zho_Hans"
TARGET_LANGUAGE = "eng_Latn"
```
Then run it. That's all~
