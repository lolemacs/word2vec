from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
import argparse

from models import *
from dataset import *
from sampler import *

parser = argparse.ArgumentParser(description='Trains word2vec with negative sampling', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='book.txt', help='Path to data text file (a text file containing some text)')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train the model for')
parser.add_argument('--embedding_size', type=int, default=50, help='Size of learned embeddings (dimensionality of embedding for each word)')
parser.add_argument('--learning_rate', type=float, default=250., help='The learning rate for SGD')
parser.add_argument('--window_size', type=int, default=2, help='Size of learned embeddings (dimensionality of embedding for each word)')
parser.add_argument('--min_count', type=int, default=2, help='Minimum number of occurrences for a word to be added to the vocabulary')
parser.add_argument('--batch_size', type=int, default=10000, help='Mini-batch size used for training: gradients will be averages over that many data points')
parser.add_argument('--negative_rate', type=int, default=5, help='Negative rate for negative sampling: that many negative pairs will be sampled for each positive one')
args = parser.parse_args()


class word2vec():
    """Main class of the project, which implements the word2vec algorithm. Given
    a Sampler (which in turn is created from a Dataset), word2vec creates a
    ContrastiveModel and trains it for a fixed number of epochs (passes through the
    positive pairs). The Sampler object is used to collect training samples (both 
    positive and negative pairs), which are fed into the ContrastiveModel. Its output
    is the used to compute the binary cross-entropy loss (BCELoss), and its gradients
    (which are stored internally in ContrastiveModel after loss.backward()) are used
    to train the model's parameter (the embeddings themselves) with optimizer.step().

    Args:
        dataset (Dataset): a Dataset object constructed from a text file.
        batch_size (int): size of the mini-batch. it is the number of positive pairs
            to be returned each time fetch_minibatch() is called.
        window_size (int): size of the moving window for the sampler. a window
            composed of 2*window_size+1 words will be moved along sentences of
            the text and return examples (word index pairs) for the neural network.
        negative_rate (int): number of negative pairs to be sampled for each positive
            pair. if set to 5 (for example), fetch_minibatch() will return 6*batch_size
            many pairs: batch_size many positive examples and 5*batch_size many negative
            ones.

    Example:
        >>> dataset = Dataset('data/war_and_peace.txt', 5)
        >>> sampler = Sampler(dataset, 1000, 2, 5)
        >>> w2v = word2vec(100)
        >>> w2v.train(sampler, num_epochs=5, lr=25.0)
    """
    
    
    def __init__(self, embedding_size):
        assert embedding_size > 0, "Embedding size must be positive"
    
        self.embedding_size = embedding_size


    def train(self, sampler, num_epochs, lr):
        """Method to train the embeddings (the goal of word2vec itself). It uses mini-batching
        to train the embeddings (parameters of ContrastiveModel) for num_epochs total epochs.
        At each mini-batch, it uses the loss' gradients to update the parameters with stochastic
        gradient descent.

        Args:
            sampler (Sampler): a Sampler object constructed from a Dataset,
                which contains the training data itself (text data in index format).
            num_epochs (int): number of epochs to train the model for. each epoch
                consists of a full pass through the positive pairs.
            lr (float): the learning rate for SGD. the update equation is
                                    w_{t+1} = w_t - lr * w_t.grad
                where w_t.grad is the gradient of the BCE loss w.r.t. w at point w_t
        """
        
        assert isinstance(sampler, Sampler), "the 'sampler' argument must be an object of the Sampler class"
        assert num_epochs > 0, "Number of epochs must be positive"
        assert lr > 0, "Learning rate must be positive"
    
        self.model = ContrastiveModel(sampler.vocab_size, self.embedding_size)

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.1)

        for epoch in range(1,num_epochs+1):
            for i in range(len(sampler.pairs)//sampler.batch_size):
                x, y = map(torch.tensor, sampler.fetch_minibatch(i))

                optimizer.zero_grad()
                outputs = self.model(x[:,0], x[:,1])
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

            print("Epoch %s: Loss %s"%(epoch, loss.item()))


    def generate_normalized_embs(self):
        """Pre-computes a row-normalized version of the embedding matrix. This
        way computing closest words (where closeness is defined as the angle)
        becomes just dot products. The result is stored as self.normalized_emb_matrix
        """
        
        embedding_matrix = self.model.embedder1.weight.data.numpy()
        row_norms =  np.sqrt((embedding_matrix**2).sum(axis=1, keepdims=True))
        self.normalized_emb_matrix = embedding_matrix / row_norms


    def closest_words(self, words, num_closest):
        """For each word in the list 'words' (each element being a string), prints
        the closest words (num_closest many) in the embedded space, where by closest
        we mean angle-wise. First, generate_normalized_embs() performs row-wise normalization
        of the embedding matrix, so each embedding has unit norm. Then computing angles becomes
        just dot products, which makes is extremely easier to find the top closest words (we can
        use plain matrix multiplications to find angles with all words in the vocabulary).
        """
    
        self.generate_normalized_embs()
        
        for word in words:
            if word in dataset.word_to_index:
                word_index = dataset.word_to_index[word]
                
                # picks the normalized embedding for the query word
                normalized_word_embedding = self.normalized_emb_matrix[word_index]
                
                # since all embeddings are normalized, a matrix multiplication computes
                # the angle between the query word and all other words in the vocabulary
                angles = np.dot(self.normalized_emb_matrix,normalized_word_embedding)
                
                # manually set the angle with itself to -infinity since we don't want to consider it
                angles[word_index] = -np.inf
                
                # finds the top num_closest indices and values, and reverses it so it's in ascending order
                closest_indexes = np.argpartition(angles, -num_closest)[-num_closest:][::-1]
                closest_words = map(lambda idx: (dataset.index_to_word[idx], angles[idx]), closest_indexes)
                print("Closest words for %s"%word.upper())
                for close_word in closest_words:
                    print("    %s: similarity %s"%(close_word[0].upper(), close_word[1]))
            else:
                print("Word %s not in vocabulary"%word)


dataset = Dataset(args.data_path, min_count=args.min_count)
sampler = Sampler(dataset, batch_size=args.batch_size, window_size=args.window_size, negative_rate=args.negative_rate)
w2v = word2vec(embedding_size=args.embedding_size)
w2v.train(sampler, num_epochs=args.epochs, lr=args.learning_rate)
w2v.closest_words(words=['harry', 'eat', 'dog', 'paper', 'snape'], num_closest=3)
