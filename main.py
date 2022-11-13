import argparse

from utils import *
from model import OurModel
from tokenizer import Tokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--min_count', type=int, default=5, help='word2vec embedding min count')
parser.add_argument('--tokenizer', type=str, defalut='bert')
args = parser.parse_args()
print(args)

dataset = []
tokenizer = Tokenizer(args.tokenizer)
word2vec = make_word2vec(tokenize_alldata(dataset, tokenizer), args.min_count)
model = OurModel()
