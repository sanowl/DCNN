import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict

class DCNNEncoder(nn.Module):
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
    

class  Generator(nn.Module):
    def __init__(self, latent_dim: int, vocab_size: int, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        
        self.fc1 = nn.Linear(latent_dim,hidden_dim * 4 )
        self.fc2 = nn.Linear(hidden_dim*4,hidden_dim*8)
        self.fc3 = nn.Linear(hidden_dim * 8,vocab_size)

        self.dropout  = nn.Dropout(0.3)

    def forward(self,z):
        x = F.relu(self.norm1(self.fc1(z)))
        x = self.dropout(x)
        x = F.relu(self.norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x,dim =-1)
    

class Discriminator(nn.Module):
    def __init__(self,vocab_size:int,hidden_dim: int = 256):
        super().__init__()

        self.encoder = DCNNEncoder(vocab_size,hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*4,hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        features = self.encoder(x)
        return self.classifier(features)
    

class CRFLayer(nn.Module):
    def __init__ (self,num_tags:int):
        super().__init__()
        self.num_tags = num_tags
        self.transtions = nn.Parameter(torch.randn(num_tags,num_tags))

    def forward(self,emissons,tags,mask = None):
        # compute the CRF loss
        return self._compiled_loss(emissons,tags,mask)
    
    def decode(self,emissinos,mask = None):
        # Viterbi decoding
        return self.viterbi_decode(emissinos,mask)
    

class HighPerformanceNLP(nn.Module):
    def __init__(self, vocab_size: int, num_tags: int, hidden_dim: int = 256):
        super().__init__()
        
        # Main components
        self.encoder = DCNNEncoder(vocab_size, hidden_dim)
        self.generator = Generator(hidden_dim, vocab_size)
        self.discriminator = Discriminator(vocab_size)
        self.crf = CRFLayer(num_tags)
        
        # Task-specific heads
        self.classification_head = nn.Linear(hidden_dim * 4, num_tags)
        self.language_model_head = nn.Linear(hidden_dim * 4, vocab_size)
        
    def forward(self, x, task='classification'):
        features = self.encoder(x)
        
        if task == 'classification':
            logits = self.classification_head(features)
            return logits
        elif task == 'generation':
            return self.generator(features)
        elif task == 'discrimination':
            return self.discriminator(x)
        else:
            raise ValueError(f"Unknown task: {task}")

# Training utilities
class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int]):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = torch.tensor([self.vocab.get(word, 0) for word in self.texts[idx].split()])
        label = torch.tensor(self.labels[idx])
        return text, label

def train_model(model, train_loader, optimizer, epochs=10, device='cuda'):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f'Epoch: {epoch}, Loss: {total_loss/len(train_loader)}')



