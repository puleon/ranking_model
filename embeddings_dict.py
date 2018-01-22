# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import copy
import numpy as np
from gensim.models.wrappers import FastText
from gensim.models import Word2Vec

class EmbeddingsDict(object):
    """The class provides embeddings using fasttext model.

    Attributes:
        tok2emb: a dictionary containing embedding vectors (value) for tokens (keys)
        embedding_dim: a dimension of embeddings
        opt: given parameters
        fasttext_model_file: a file containing fasttext binary model
    """

    def __init__(self, opt):
        """Initialize the class according to given parameters."""

        self.tok2emb = {}
        self.opt = copy.deepcopy(opt)
        self.embedding_dim = self.opt['embedding_dim']
        self.embeddings_model_file = self.opt['embeddings_model_file']
        self.load_items()
        self.embeddings = self.opt["embeddings"]

        if self.embeddings == "fasttext":
            self.embeddings_model = FastText.load_fasttext_format(self.embeddings_model_file)
        elif self.embeddings == "word2vec":
            self.embeddings_model = Word2Vec.load(self.embeddings_model_file)

        assert(self.embeddings_model["I"].shape[0] == self.embedding_dim),\
            'The dimensionality of embeddings does not equal to the embedding_dim parameter. Exit.'

    def add_items(self, sentence_li):
        """Add new items to the tok2emb dictionary from a given text."""
        dummy_emb = list(np.zeros(self.embedding_dim))
        for sen in sentence_li:
            for tok in sen:
                if self.tok2emb.get(tok) is None:
                    try:
                        self.tok2emb[tok] = self.embeddings_model[tok]
                    except:
                        self.tok2emb[tok] = dummy_emb

    def save_items(self):
        """Save the dictionary tok2emb to the file."""
        fname = self.opt.get('embeddings_dict')
        if fname is not None and not os.path.isfile(fname):
            f = open(fname, 'w')
            string = '\n'.join([el[0] + ' ' + self.emb2str(el[1]) for el in self.tok2emb.items()])
            f.write(string)
            f.close()

    def emb2str(self, vec):
        """Return the string corresponding to given embedding vectors"""

        string = ' '.join([str(el) for el in vec])
        return string

    def load_items(self):
        """Initialize embeddings from the file."""

        fname = self.opt.get('embeddings_dict')
        if fname is not None:
            if not os.path.isfile(fname):
                print('There is no %s file provided. Initializing new dictionary.' % fname)
            else:
                print('Loading existing dictionary  from %s.' % fname)
                with open(fname, 'r') as f:
                    for line in f:
                        values = line.rsplit(sep=' ', maxsplit=self.embedding_dim)
                        assert(len(values) == self.embedding_dim + 1)
                        word = values[0]
                        coefs = np.asarray(values[1:], dtype='float32')
                        self.tok2emb[word] = coefs

