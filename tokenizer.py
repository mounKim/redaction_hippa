from utils import *
from transformers import BertTokenizer, PreTrainedTokenizer


class Tokenizer(PreTrainedTokenizer):
    def __init__(self, tokenizer_name, lower=True, all_zero=True):
        super(Tokenizer, self).__init__()
        if tokenizer_name == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.lower = lower
        self.all_zero = all_zero

    def tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize(token_normalization(text))

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)
