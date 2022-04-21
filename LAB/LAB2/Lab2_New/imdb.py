from config import *
from torch.utils.data import DataLoader, Dataset
import os, random
from collections import Counter
import nltk
import numpy as np

class IMDBMovieReviews():
    def __init__(self, root):
        self.root = root
        return

    def get_data(self):
        def extract_data(dir, split):
            data = []
            for label in ("pos", "neg"):
                label_dir = os.path.join(dir, "aclImdb", split, label)
                files = sorted(os.listdir(label_dir))
                for file in files:
                    filepath = os.path.join(label_dir, file)
                    with open(filepath, encoding="UTF-8") as f:
                        data.append({L_RAW: f.read(), L_LABEL: label})
            return data

        train_data = extract_data(self.root, "train")
        test_data = extract_data(self.root, "test")
        return train_data, test_data

    def split_data(self, train_data, num_split=2000):
        random.shuffle(train_data)
        return train_data[:-num_split], train_data[-num_split:]

    def tokenize(self, data, max_seq_len=MAX_SEQ_LEN):
        for review in data:
            review[L_TOKENS] = []
            for sent in nltk.sent_tokenize(review[L_RAW]):
                review[L_TOKENS].extend(nltk.word_tokenize(sent))
            if max_seq_len >= 0:
                review[L_TOKENS] = review[L_TOKENS][:max_seq_len]

    def create_vocab(self, data, unk_threshold=UNK_THRESHOLD):
        counter = Counter(token for review in data for token in review[L_TOKENS])
        self.vocab = {token for token in counter if counter[token] > unk_threshold}
        token_to_idx = {PAD: 0, UNK: 1}
        for token in self.vocab:
            token_to_idx[token] = len(token_to_idx)
        return token_to_idx

    def get_embeds(self, token_to_index_mapping, token_to_glove, dim):
        weights_matrix = np.zeros((len(token_to_index_mapping), dim))
        indices_found = []

        for word, i in token_to_index_mapping.items():
            if word in token_to_glove.keys():
                indices_found.append(i) # This gradient of these indices will get zero'd out later depending on config
            weights_matrix[i] = token_to_glove.get(word, np.random.normal(size=(dim, )))
        return indices_found, weights_matrix

    def apply_vocab(self, data, token_to_idx):
        for review in data:
            review[L_TOKENS] = [token_to_idx.get(token, token_to_idx[UNK]) for token in review[L_TOKENS]]


    def apply_label_map(self, data, label_to_idx):
        for review in data:
            review[L_LABEL] = label_to_idx[review[L_LABEL]]

class SentimentDataset(Dataset):
    def __init__(self, data, pad_idx):
        data = sorted(data, key=lambda review: len(review[L_TOKENS]))
        self.texts = [review[L_TOKENS] for review in data]
        self.labels = [review[L_LABEL] for review in data]
        self.pad_idx = pad_idx

    def __getitem__(self, index):
        return [self.texts[index], self.labels[index]]

    def __len__(self):
        return len(self.texts)

    def collate_fn(self, batch):
        def tensorize(elements, dtype):
            return [torch.tensor(element, dtype=dtype) for element in elements]

        def pad(tensors):
            max_len = max(len(tensor) for tensor in tensors)
            padded_tensors = [
                F.pad(tensor, (0, max_len - len(tensor)), value=self.pad_idx) for tensor in tensors
            ]
            return padded_tensors

        texts, labels = zip(*batch)
        return [
            torch.stack(pad(tensorize(texts, torch.long)), dim=0),
            torch.stack(tensorize(labels, torch.long), dim=0),
        ]