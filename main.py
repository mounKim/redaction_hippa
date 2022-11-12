import argparse

from model import OurModel
from tokenizer import Tokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--tokenizer', type=str, defalut='bert')
args = parser.parse_args()
print(args)

model = OurModel()
tokenizer = Tokenizer(args.tokenizer)
