import torch
import torch.nn as nn
import torch.nn.init as init

class ContrastiveModel(nn.Module):
    """PyTorch Module that implementes the word2vec model where
    model(w1,w2) = sigma(<e(w1),e(w2)>), which is the main term of the 
    binary cross-entropy loss we want to minimize. In practice we use
    two embedding mappings e1 and e2 instead, where e1 is used to embed
    the center word (of the window) and e2 embeds the context word (some
    other word, not in the window's center). The two embedding layers
    contain trainable parameters that are internally updated via
    gradient-descent methods with PyTorch.
    

    Args:
        vocabulary_size (int): number of words for which the model learns embeddings.
        emb_size (int): size of the learned embeddings. each nn.Embedding() Module
            performs a mapping from a word index (0, ..., vocabulary_size) to a vector
            of emb_size dimensions.

    Example:
        >>> dataset = Dataset('data/war_and_peace.txt', 5)
    """
    
    
    def __init__(self, vocabulary_size, emb_size):
        super(ContrastiveModel, self).__init__()
        assert vocabulary_size > 2, "Vocabulary size must be greater than 2"
        assert emb_size > 0, "Embedding size must be positive"
        
        self.embedder1 = nn.Embedding(vocabulary_size, emb_size)
        self.embedder2 = nn.Embedding(vocabulary_size, emb_size)
        self.sigmoid = nn.Sigmoid()
        
        init.uniform_(self.embedder1.weight, a=-0.5/emb_size, b=0.5/emb_size)
        init.constant_(self.embedder2.weight, 0)
        
        
    def forward(self, word1, word2):
        """Computes sigma(<e1(w1),e2(w2)>)

        Args:
            word1 (int): index of the first word in the pair.
            word2 (int): index of the second word in the pair.

        Example:
            >>> model = ContrastiveModel(5000, 100)
            >>> print(model.forward(0,1))
        """
    
        embedding1 = self.embedder1(word1)
        embedding2 = self.embedder2(word2)
        return self.sigmoid(torch.sum(embedding1*embedding2, dim=1))
