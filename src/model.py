""" MT Model Class """
import os
import logging
from time import perf_counter

import torch
from ctranslate2 import Translator
from transformers import AutoTokenizer

# pylint: disable=too-few-public-methods
class MTModel:
    """MT Model instance"""

    def __init__(self, model_dir: str, source_language: str, target_language: str):
        """
        model_dir (str):
            Directory where your model and tokenizer files are stored
            File directory should look like this:
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

        source_language (str): Language to translate from, uses FLORES-200 language code
        target_language (str): Language to translate into, uses FLORES-200 language code
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info("Running on device: %s", self.device)

        logging.info("Loading model...")
        model_load_start = perf_counter()

        self.source_language = source_language
        self.target_language = target_language

        tokenizer_path = os.path.join(model_dir, "tokenizer")
        assert os.path.exists(tokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_dir, "tokenizer"),
            src_lang=self.source_language,
            tgt_lang=self.target_language,
        )

        model_path = os.path.join(model_dir, "model")
        assert os.path.exists(model_path)
        self.model = Translator(model_path, device=self.device)

        model_load_end = perf_counter()
        logging.info(
            "Model loaded. Elapsed time: %s", model_load_end - model_load_start
        )

    def _translate(self, input_text: str) -> str:
        """Takes in a text and translate Language X -> English
        referenced: https://opennmt.net/CTranslate2/guides/transformers.html#nllb
        """

        source = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(input_text))

        results = self.model.translate_batch([source], target_prefix=[[self.target_language]])
        target = results[0].hypotheses[0][1:]

        output_text = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(target))

        return output_text

    # pylint: disable=unused-argument,no-member
    def translate(self, text: str) -> str:
        """ Wraps the internal method. Replace this method with a wrapper for your service call"""

        infer_start = perf_counter()
        output_text = self._translate(text)
        infer_end = perf_counter()
        logging.info("Inference elapsed time: %s", infer_end - infer_start)

        return output_text
