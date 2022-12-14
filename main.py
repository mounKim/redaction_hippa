import argparse

from utils import *
from train import *
from model import OurModel
from tokenizer import Tokenizer
from dataset import ClinicalDataset
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--rigid_labeling', type=bool, default=False, help='if false, labeling only name/date/location')
parser.add_argument('--batch', type=int, default=20)
parser.add_argument('--dataset_path', type=str, default='data')
parser.add_argument('--device', type=str, default='cpu', help='cpu or gpu')
parser.add_argument('--embedding_model', type=str, default='glove.txt', help='default means training word2vec')
parser.add_argument('--word2vec_min_count', type=int, default=10, help='word2vec embedding min count')
parser.add_argument('--tokenizer', type=str, default='bert')
args = parser.parse_args()
print(args)

i2b2_2014_texts_train, i2b2_2014_labels_train = preprocess_i2b2_2014(
    os.path.join(args.dataset_path, 'i2b2-2014/train'), args.rigid_labeling)
i2b2_2014_texts_test, i2b2_2014_labels_test = preprocess_i2b2_2014(
    os.path.join(args.dataset_path, 'i2b2-2014/test/test_answer'), args.rigid_labeling)

label = [i2b2_2014_labels_train]
labeled_data = [i2b2_2014_texts_train]
unlabeled_data = [i2b2_2014_texts_train]
val_data = [i2b2_2014_texts_test[:250]]
val_label = [i2b2_2014_labels_test[:250]]
test_data = [i2b2_2014_texts_test[250:]]
test_label = [i2b2_2014_labels_test[250:]]

tokenizer = Tokenizer(args.tokenizer)
if args.embedding_model == '':
    embedding = make_word2vec(tokenize_alldata(sum(unlabeled_data, []), tokenizer), args.word2vec_min_count)
else:
    embedding = load_glove_model(args.embedding_model)

model = OurModel(args, reduction='token_mean').to(args.device)
print('model is loaded')

if args.mode == 'train':
    train_dataset = ClinicalDataset(sum(labeled_data, []), sum(label, []), tokenizer, embedding)
    val_dataset = ClinicalDataset(sum(val_data, []), sum(val_label, []), tokenizer, embedding)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, collate_fn=my_collate)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch, collate_fn=my_collate)
    train(model, args, train_dataloader, val_dataloader)
else:
    test_dataset = ClinicalDataset(sum(test_data, []), sum(test_label, []), tokenizer, embedding)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, collate_fn=my_collate)
    predict(model, args, test_dataloader)
