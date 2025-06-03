# gensim -- > numpy

import numpy as np
import struct

class Word2Vec:
    def init_(self, filepath):
        print("Loading Word2Vec ... ")
        self.word2vec = {}
        with open(filepath, 'rb') as f:
            header = f.readline()
            vocab_size, vector_size = map(int, header.split())
            self.vector_size = vector_size

            for _ in range(vocab_size):
                word = b''
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch != b'\n':
                        word += ch
                word = word.decode('utf-8', errors='ignore')
                vec = np.frombuffer(f.read(4 * vector_size), dtype=np.float32)
                self.word2vec[word] = vec

        print(f"Loaded {len(self.word2vec)} words.")

        self.words = list(self.word2vec.keys())
        self.embeddings = np.array([self.word2vec[w] for w in self.words])
        self.embeddings /= np.linalg.norm(self.embeddings, axis=1, keepdims=True)

    def most_similar(self, positive, negative=[], topn=10):
        vec = sum(self.word2vec[w] for w in positive) - sum(self.word2vec[w] for w in negative)
        vec /= np.linalg.norm(vec)
        
        sims = self.embeddings.dot (vec)
        best_idx = np.argsort(-sims)[:topn+ len(positive) + len(negative)]
        results = []

        for idx in best_idx:
            word = self.words[idx]
            if word not in positive + negative:
                results.append((word, float(sims[idx])))
            if len(results) == topn:
                break
        return results

    def neighbors(self, word, topn=10): #기능
        return self.most_similar([word], topn=topn)
            
