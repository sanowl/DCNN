import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict

class DONEncoder(nn.Module):
    def __init__(self,vocab_size:int,embed_dim: int = 300,hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim)

        # deep cnn layers
        self.conv1 = nn.Conv1d(embed_dim,hidden_dim,kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(hidden_dim,hidden_dim=1,kernel_size=3,padding=1)
        self.conv3 = nn.Conv1d(hidden_dim=2,hidden_dim=4,kernel_size=3,padding=1)

        # normlaztion layer
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim=2)
        self.norm3 = nn.BatchNorm1d(hidden_dim=4)

        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout1d(0.5)

    def forward(self,x):
        # x shape[batch_size,squence_lenghth]
        x = self.embedding(x)
        x = x.transpose(1,2)

        # applying cnn layer with residual connection
        x1 = self.pool(F.relu(self.norm1(self.conv1(x))))
        x2 = self.pool(F.relu(self.norm2(self.conv2(x1))))
        x3 = self.global_pool(F.relu(self.norm3(self.conv3(x2))))

        x =x3.sequeeze(-1)
        x = self.dropout(x)

        return x
