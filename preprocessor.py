from transformers import BertTokenizer, BatchEncoding
from ray import serve

from torch.utils.data import DataLoader

import re


@serve.deployment(num_replicas=1)
class TextPreprocessor:

    def __init__(self):
        self.bert_tokenizer: BertTokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-uncased")

    async def preprocess(self, data: list[str]) -> dict:

        def clean_text(string: str) -> str:
            converted_string = re.sub(
                pattern=r"[^a-zA-Z가-힣0-9,.!?'()#$%^&\"]",
                repl=" ",
                string=string,
            )
            normalized_string = re.sub(
                pattern=r"\s+",
                repl=" ",
                string=converted_string,
            )

            return normalized_string.strip()

        cleaned_data: list[str] = list(map(clean_text, data))

        preprocessed_data = self.bert_tokenizer(
            text=cleaned_data,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )

        return preprocessed_data.data

