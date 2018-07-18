import torch
import torch.nn as nn
import torchvision
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk import sent_tokenize, word_tokenize
from collections import Counter
from itertools import chain
import time
import random
import argparse

from models import *
from dataset import *
from sampler import *

parser = argparse.ArgumentParser(description='Train word2vec', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='book.txt', help='Path to data text file (a text file containing some text)')
parser.add_argument('--mode', type=str, default='negative_sampling', choices=['negative_sampling', 'softmax'], help='Word2vec training mode: either "negative_sampling" or "softmax"')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train the model for')
parser.add_argument('--embedding_size', type=int, default=50, help='Size of learned embeddings (dimensionality of embedding for each word)')
parser.add_argument('--window_size', type=int, default=2, help='Size of learned embeddings (dimensionality of embedding for each word)')
parser.add_argument('--min_count', type=int, default=2, help='Minimum number of occurrences for a word to be added to the vocabulary')
parser.add_argument('--batch_size', type=int, default=10000, help='Mini-batch size used for training: gradients will be averages over that many data points')
parser.add_argument('--negative_rate', type=int, default=5, help='Negative rate for negative sampling: that many negative pairs will be sampled for each positive one')
args = parser.parse_args()


class word2vec():
    def __init__(self, mode, embedding_size):
        self.mode = mode
        self.embedding_size = embedding_size

    def train(self, sampler, num_epochs, lr):
        self.model = ContrastiveModel(sampler.vocab_size, self.embedding_size)

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.1)

        for epoch in range(num_epochs):
            for i in range(len(sampler.pairs)//sampler.batch_size):
                x, y = sampler.fetch_minibatch(i)

                optimizer.zero_grad()
                outputs = self.model(x[:,0], x[:,1])

                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                print "Loss: %s"%(loss.item())

        quit()

        self.calc_normalized_embs()
        print "Harry: ", self.closest_words("harry")
        print "Magic: ", self.closest_words("magic")
        print "Wizard: ", self.closest_words("wizard")
        print "Hagrid: ", self.closest_words("hagrid")
        print "Dumbledore: ", self.closest_words("dumbledore")






    def calc_normalized_embs(self):
        embs = self.model.state_dict()['emb.weight']
        embs = embs.cpu().numpy()
        #embs2 = self.model.state_dict()['emb2.weight']
        #embs2 = embs2.cpu().numpy()
        #embs = (embs + embs2)/2
        self.norm_embs = embs / np.sqrt((embs**2).sum(axis=1))[:, np.newaxis]

    def closest_words(self, word, k=5):
        idx = self.vocab[word]
        distances = np.dot(self.norm_embs,self.norm_embs[idx])
        distances[idx] = -np.inf
        top_n = np.argpartition(distances, -k)[-k:]
        return map(lambda i: (self.inv_vocab[i], distances[i]), top_n)



"""
def plot_embs(model, epoch):
    embs = model.state_dict()['emb.weight']
    embs = embs.cpu().numpy()
    plt.scatter(embs[:,0], embs[:,1], color=['red','green','blue', 'yellow'])
    plt.ylabel('some numbers')
    plt.show()
    plt.savefig('plot/embs_%s.png'%epoch)
"""



dataset = Dataset(args.data_path, min_count=args.min_count)
sampler = Sampler(dataset, mode=args.mode, batch_size=args.batch_size, negative_rate=args.negative_rate)
w2v = word2vec(mode=args.mode, embedding_size=args.embedding_size)
w2v.train(sampler, num_epochs=args.epochs, lr=0.025*args.batch_size)





