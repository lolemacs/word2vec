from __future__ import print_function

from dataset import Dataset
import numpy as np

class Sampler():
    """Class to create a word context sampler from a dataset. This sampler
    is used to sample word pairs for negative sampling, when training
    a neural network to learn word embeddings (fixed-dimensional vectors for words).
    
    The method fetch_minibatch() is the only one supposed to be used externally.

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
        >>> print(sampler.fetch_minibatch(0))
    """
    
    
    def __init__(self, dataset, batch_size, window_size=2, negative_rate=1):
        assert isinstance(dataset, Dataset), "dataset must be an object of the Dataset class"
        assert batch_size > 0, "Batch size has to be at least 1"
        assert window_size > 0, "Window size has to be at least 1"
        assert negative_rate >= 0, "Negative rate cannot be negative"
        
        self.dataset = dataset
        self.vocab_size = dataset.vocab_size
        self.window_size = window_size
        self.batch_size = batch_size
        self.negative_rate = negative_rate
        
        self.precompute_sampling_probabilities()
        self.generate_context()


    def precompute_sampling_probabilities(self):
        """Pre-computes probabilities for negative sampling. In negative sampling,
        we have a parameter 'nr' (negative rate) and we sample nr negative pairs
        for each positive pair: a positive pair is a pair of words that appear close
        together in the text, while a negative pair is a set of two 'randomly 
        chosen' words.
        
        In traditional negative sampling, we sample a word w with probability 
        p(w) = count(w)/sum_w' count(w'), which is the word's frequency
        in the text. Here we use a smoothing technique that consists of computing
        count(w)^3/4 for each word and then normalizing. 
        
        This method pre-computes these probabilities to be used during the 
        sampling itself, and populates self.sampling_probs.
        """
        
        assert isinstance(self.dataset.index_counts, dict), "Dataset does not seem to have a dict attribute dataset.index_counts"
        
        indices, index_counts = self.dataset.index_counts.keys(), self.dataset.index_counts.values()
        self.sampling_indexes = np.array(indices)
        corrected_counts = np.array(index_counts, dtype='float32') ** 0.75
        self.sampling_probs = corrected_counts / corrected_counts.sum()


    def generate_context(self):
        """Pre-generates set of positive word pairs to be used for training later.
        Positive pairs are defined by a set of two words w1, w2 such that the distance
        between the two in the text is at most the window size. So for a sentence
        s = w1 w2 w3 w4 and window size = 1, the (unordered) positive pairs are (w1,w2),
        (w2,w3) and (w3,w4).
        
        This is implemented by sliding a moving window through each index sequence
        and storing pairs (wc,wp) where wc is the center word (in the middle of the window)
        and wp is other word in the window (but not wc).
        """
    
        pairs = []
        for sequence in self.dataset.index_sequences:
            if len(sequence) < self.window_size: continue
            for i in range(len(sequence)):
                center_word = sequence[i]
                
                # extracts all words at least window_size-close to the center word, respecting boundaries
                context_window = sequence[max(0,i-self.window_size) : min(len(sequence),i+self.window_size+1)]
                
                # adds (center word, context word) to the list of pairs
                for context_word in context_window:
                    if context_word != center_word: pairs.append([center_word, context_word])
                    
        self.pairs = np.asarray(pairs)
        np.random.shuffle(self.pairs)


    def fetch_minibatch(self, batch_index):
        """Fetches a mini batch given by batch_index (0, 1, 2, ..., len(self.pairs)/batch_size),
        which includes training pairs along with their labels. For negative sampling,
        this returns (negative_rate+1)*batch_size pairs, which include batch_size
        positive pairs (label 1) and negative_rate*batch_size negative pairs (label 0).
        
        Args:
            batch_index (integer): the mini batch index. this should be an integer
                between 0 and len(pairs)/batch_size.
        """
    
        assert batch_index >= 0, "Batch index cannot be negative"
    
        start, end = batch_index*self.batch_size, (batch_index+1)*self.batch_size
        positive_pairs = self.pairs[start:end]
        batch_pairs = positive_pairs

        # performs negative_rate number of negative samples, each collecting batch_size negative pairs
        for i in range(self.negative_rate):
            negative_indices = np.random.choice(self.sampling_indexes, p=self.sampling_probs, size=self.batch_size)
            negative_pairs = positive_pairs.copy()
            negative_pairs[:,1] = negative_indices
            batch_pairs = np.concatenate((batch_pairs, negative_pairs))

        # sets the labels accordingly: 1 for positive pairs and 0 for negative ones
        batch_labels = np.zeros(batch_pairs.shape[0], dtype='float32')
        batch_labels[:positive_pairs.shape[0]] = 1

        return batch_pairs, batch_labels


if __name__ == '__main__':
    from dataset import *
    
    dataset = Dataset('book.txt', 5)
    sampler = Sampler(dataset, 8, 2, 1)
    x, y = sampler.fetch_minibatch(0)
    print("Training pairs: ", x)
    print("Training labels: ", y)
