from config import *
import numpy as np
import os
class GloVeWordEmbeddings():
    def __init__(self, glove_file_path, num_dims):
        self.num_dims = num_dims


        if not os.path.exists(glove_file_path):
            print("Error! Not a valid glove path")
            return
        
        self.token_to_embedding = {
            PAD: np.random.normal(size=(num_dims, )),
            UNK: np.random.normal(size=(num_dims, ))
        }

        with open(glove_file_path, 'r', encoding="utf-8") as f:
            for i, line in enumerate(f):
                values = line.split()

                # create a dict of word -> positions
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                
                self.token_to_embedding[word] = vector
    
    def get_token_to_embedding(self):
        return self.token_to_embedding
    
    def get_num_dims(self):
        return self.num_dims

    def _get_cosine_similarity(self, vecA: np.array, vecB: np.array):
        return np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))

    def _get_closest_words(self, embedding):
        return sorted(self.token_to_embedding.keys(), key=lambda w: self._get_cosine_similarity(self.token_to_embedding[w], embedding), reverse=True)
    
    def _get_embedding_for_word(self, word: str) -> np.array:
        if word in self.token_to_embedding.keys():
            return self.token_to_embedding[word]
        return np.array([])

    def get_x_closest_words(self, word, num_closest_words=1) -> list: 

        embedding = self._get_embedding_for_word(word)
        if embedding.size == 0:
            print(f"{word} does not exist in the embeddings.")
            return []
        closest_words = self._get_closest_words(embedding)
        for w in [word, PAD, UNK]: closest_words.remove(w)

        return closest_words[:num_closest_words]
    
    def get_word_analogy_closest_words(self, w1, w2, w3, num_closest_words=1):
        e1 = self._get_embedding_for_word(w1)
        e2 = self._get_embedding_for_word(w2)
        e3 = self._get_embedding_for_word(w3)

        if e1.size == 0 or e2.size == 0 or e3.size == 0:
            print(f"{w1}:{e1.size}  {w2}:{e2.size}  {w3}:{e3.size}")
            return []

        embedding = e2 - e1 + e3
        closest_words = self._get_closest_words(embedding)
        for w in [w1, w2, w3, PAD, UNK]: closest_words.remove(w) 
        return closest_words[:num_closest_words]