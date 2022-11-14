import argparse

from utils import *
from model import OurModel
from tokenizer import Tokenizer
from dataset import ClinicalDataset
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--word2vec_model', type=str, default='', help='default means training word2vec')
parser.add_argument('--word2vec_min_count', type=int, default=5, help='word2vec embedding min count')
parser.add_argument('--tokenizer', type=str, default='bert')
args = parser.parse_args()
print(args)

# data must be 1d list
labeled_data = []
unlabeled_data = []
tokenizer = Tokenizer(args.tokenizer)
if args.word2vec_model == '':
    word2vec = make_word2vec(tokenize_alldata(unlabeled_data, tokenizer), args.word2vec_min_count)
else:
    word2vec = Word2Vec.load(args.word2vec_model)

model = OurModel()
# dataset = ClinicalDataset(_, _, _)
