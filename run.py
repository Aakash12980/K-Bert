import click
import torch
import tokenizer
from data import ClassifierDataset
from bert import BertModel
from torch.utils.data import DataLoader
import torch.optim as optim

BATCH_SIZE = 4
EPOCH = 10
lr = 3e-5


CONTEXT_SETTINGS = dict(helo_options_names = ['-h', '--help'])

def collate_fn(batch):
    data = [data[0] for data in batch]
    label = [data[1] for data in batch]
    return dtaa, label

@click.group(CONTEXT_SETTINGS=CONTEXT_SETTINGS)
@click.version_option(version='1.0.0')
def task():
    ''' This is the documentation of the main file. This is the reference for executing this file.'''
    pass

@click.command()
@click.option('--train_path', default="./dataset/train.txt", help="Path for train file")
@click.option('--valid_path', default="./dataset/valid.txt", help="Path for validation file")
@click.option('--seed', default=423, help="Manual seed value default(423)")
def train(**kwargs):
    train_dataset = ClassifierDataset(kwargs["--train_path"])
    valid_dataset = ClassifierDataset(kwargs["--valid_path"])
    print("Dataset loaded successfully")

    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    valid_dl = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

    model = BertModel()
    optimizer = optim.Adam(model.parameters(), lr)
    
    BertModel.trainer(model, optimizer, EPOCH)
    
    

@click.command()
@click.option('--test_path', default="./dataset/test.txt", help="Path for test file")
def test(**kwargs):
    pass


def main():
    pass

if __name__ == "__main__":
    main()