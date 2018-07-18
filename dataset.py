import nltk
from nltk import sent_tokenize, word_tokenize
from collections import Counter
from itertools import chain
import os
import sys

class Dataset():
    """Class to create a pre-processed dataset from a text file. Given a path
    to a text file (containing plain text), this class assigns unique integer
    indexes to each word that appears at least min_count times in the text.
    
    No method is supposed to be called by the user. A Dataset object should
    be passed to a Sampler, which does the actual sampling of training examples.

    Args:
        data_path (string): path to the text file containing the textual dataset
            it has to be a text file containing plain text, in plain encoding
            or utf-8.
        min_count (int): minimum number of occurrences for a word to be considered.
            words that appear less than min_count times in the dataset will be
            ignored (they will not be considered during the sampling procedure
            and no embeddings will be learned for them).

    Example:
        >>> dataset = Dataset('data/war_and_peace.txt', 5)
    """
    
    
    def __init__(self, data_path, min_count=0):
        assert os.path.isfile(data_path), "No file found at %s"%data_path
        assert min_count >= 0, "Min count cannot be negative"
        
        self.min_count = min_count
        
        self.generate_word_sequences(data_path)
        self.generate_index_sequences()


    def generate_word_sequences(self, data_path):
        """Populates self.word_sequences from the text dataset at data_path.
        
        After this method executes, self.word_sequences will be a list of lists of
        words. That is, self.word_sequences[i][j] will be the j'th word (a string)
        of the i'th sentence of the text at data.path
        
        Arguments:
            data_path (string): path to the text file containing the textual dataset
                it has to be a text file containing plain text, in plain encoding
                or utf-8.
        """
    
        try:
            text = open(data_path, 'rU').read().lower()
            text = text.decode('utf8')
        except IOError as e:
            print("Error reading file: %s / %s"%(e.errno, e.strerror))
            sys.exit()
        except:
            print("Unexpected error:", sys.exc_info()[0])
            sys.exit()
    
        # uses tokenizers from NLTK (sent_tokenize splits a text (string) into a list
        # of sentences (strings), and word_tokenize splits a sentence (string) into a 
        # list of words (strings).
        print("Tokenizing sentences")
        self.word_sequences = [word_tokenize(sentence) for sentence in sent_tokenize(text)]
        
        assert len(self.word_sequences) > 0, "No word sequences were extracted, re-check text file"
        print("  Found %s total sentences"%len(self.word_sequences))


    def generate_index_sequences(self):
        """Populates some attributes, including index sequences used for training and:
        
            self.word_counts: a (word:count) dictionary where count is the number
                of occurrences in of word in the text. does not contain words
                that occur less than min_count times.
            self.word_to_index: a (word:index) dictionary where index is a unique
                integer index assigned to the word.
            self.index_sequences: the 'actual' dataset, in the sense that training
                examples are sampled from it. it is a list of index lists.
            self.vocab_size: integer, number of words in the vocabulary.
            self.index_to_word: (index:word) dictionary, inverse of self.word_to_index

        First, the vocabulary is built by assigning indexes (0, 1, 2, etc) to each
        different word in the text data (ignoring the ones that appear less than
        min_count times).
        
        Then, this method populates self.index_sequences: it is the same as
        self.word_sequences, but with each word replaced by its index. It is 
        a list of int lists. self.index_sequences[i][j] will be the index of
        the j'th word (a integer) of the i'th sentence of the text data
        """
        
        print("Building vocabulary and index sequences")
        
        # uses Counter to count how many times each word appears in the text
        # then filters out the words that are not alphabetic or occur less than min_count times
        # the so-called 'vocabulary' is the set of the words that are not filtered out
        word_counts = Counter(chain(*self.word_sequences))
        self.word_counts = {word : count for word, count in word_counts.iteritems() if (count >= self.min_count and word.isalpha())}
        
        # assigns a unique integer index to each word in the vocabulary
        indexes = range(len(self.word_counts))
        self.word_to_index = {word : index for word, index in zip(self.word_counts, indexes)}
            
        # creates anonymous functions for big call to map word sequences to index sequences
        filter_words_func = lambda word: word in self.word_to_index # filters out words not in the vocab
        word_to_index_func = lambda word: self.word_to_index[word] # maps a word to its integer index
        
        # this function maps a word sequence (list of words) into a index sequence (list of integer indices)
        # it also filters out words that are not in the vocabulary and thus have no associated index
        sentence_to_index_func = lambda sentence: map(word_to_index_func, filter(filter_words_func, sentence))

        # map each word_seq to index_seq, and filters out empty sequences (ones whose all words were filtered out)
        self.index_sequences = map(sentence_to_index_func, self.word_sequences)
        self.index_sequences = filter(len, self.index_sequences)

        # stores vocabulary size and generates index -> word dictionary
        self.vocab_size = len(self.word_to_index)
        self.index_counts = {self.word_to_index[word] : count for word, count in self.word_counts.iteritems()}
        self.index_to_word = {index : word for word, index in self.word_to_index.iteritems()}

        assert self.vocab_size > 0, "No words in vocabulary. Did you set min_count too large?"
        print("  Vocabulary has %s words total"%self.vocab_size)


if __name__ == '__main__':
    dataset = Dataset('book.txt', 5)
    print(' '.join(dataset.word_sequences[0]))
    print(dataset.word_sequences[0])
    print(dataset.index_sequences[0])
