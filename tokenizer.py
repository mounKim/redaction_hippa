from transformers import BertTokenizer, PreTrainedTokenizer


class Tokenizer(PreTrainedTokenizer):
    def __init__(self, tokenizer_name, lower=True, all_zero=True):
        super(Tokenizer, self).__init__()
        if tokenizer_name == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.lower = lower
        self.all_zero = all_zero

    def tokenize(self, text, **kwargs):
        if self.lower:
            text = text.lower()
        if self.all_zero:
            number = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
            for n in number:
                text = text.replace(n, '0')
        return self.tokenizer.tokenize(text)

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)
