import torch
import torch.nn as nn
from tokenizer import Tokenizer

# For Bert Model: numlayer = 12, attnhead = 12 hiddenlayers = 768

class BertModel(nn.Module):
    def __init__(self):
        super(BertModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=12, dim_feedforward=768)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=12)
        self.linear_layer_1 = nn.Linear(in_features=768, out_features=768)
        self.linear_layer_2 = nn.Linear(in_features=768, out_features=2)
        self.logSoftmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.BCELoss()
        self.tokenizer = Tokenizer()

    def forward(self, batch, mask):
        

    @staticmethod
    def trainer(batch, model, optimizer, dataloader, num_epoch):
        for epoch in range(num_epoch):
            print(f"Running Epoch {epoch}")

            for step, data in enumerate(dataloader):
                src_tensor, attn_mask = self.tokenizer(data[0])

