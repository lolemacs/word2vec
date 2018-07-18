import torch
import torch.nn as nn
import torch.nn.init as init

class ContrastiveModel(nn.Module):
    def __init__(self, vocabulary_size, emb_size):
        super(ContrastiveModel, self).__init__()
        self.embedder1 = nn.Embedding(vocabulary_size, emb_size)
        self.embedder2 = nn.Embedding(vocabulary_size, emb_size)
        
        init.uniform(self.embedder1.weight, a=-0.5/emb_size, b=0.5/emb_size)
        #init.uniform(self.embedder2.weight, a=-0.5/emb_dimension, b=0.5/emb_dimension)
        init.constant(self.embedder2.weight, 0)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, word1, word2):
        embedding1 = self.embedder1(word1)
        embedding2 = self.embedder2(word2)
        return self.sigmoid(torch.sum(embedding1*embedding2, dim=1))
