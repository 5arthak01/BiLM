import re
import pickle
import os

from clean import clean_text
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import Parameter, ParameterList

import gensim.downloader as gensim_downloader
from gensim.models import KeyedVectors

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%I:%M:%S %p"
)

BATCH_SIZE = 512
MAX_UNK_FREQ = 10
EMBEDDING_TYPE = "w2v"
EMBEDDING_DIM = 300 if EMBEDDING_TYPE == "w2v" else 50

# Given in question
CONTEXT_SIZE = 2
NGRAM_SIZE = CONTEXT_SIZE + 1
HIDDEN_LAYER_SIZE = 128
PROJECTION_SIZE = 64
MAX_SENT_LEN = 20

HIDDEN_LAYER_1_SIZE = 128
HIDDEN_LAYER_2_SIZE = 128

# can be anything
RIGHT_PAD_SYMBOL = "<EOS>"
LEFT_PAD_SYMBOL = "<SOS>"
UNK_TOKEN = "<UNK>"

MODEL_DIR = "./models/"
os.makedirs(MODEL_DIR, exist_ok=True)

TO_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if TO_GPU else "cpu")
CPU = torch.device("cpu")


def is_padding(word):
    return word == LEFT_PAD_SYMBOL or word == RIGHT_PAD_SYMBOL


def add_padding(text, n):
    n = max(n, 0)
    return [LEFT_PAD_SYMBOL] * n + text + [RIGHT_PAD_SYMBOL] * n


def read_data(filename):
    """
    Read data from file
    """
    with open(filename, "r") as f:
        data = f.read()
    return data


def tokenise(data):
    """
    Tokenise data, return list of sentences,
    which are lists of words
    """
    data = re.sub(r"\s+", " ", data)
    data = clean_text(data)
    # data = re.sub(r"\s+", " ", data)
    data = sent_tokenize(data)
    data = [word_tokenize(x) for x in data]
    return data


def make_vocab(data):
    """
    Make vocabulary dict from data
    """
    # data = read_data(filename)
    # data = tokenise(data)
    # data = [word for line in data for word in line]
    vocab = Counter(data)
    return vocab


def tokenise_and_pad_text(data, context_size=CONTEXT_SIZE):
    """
    Tokenise data and pad with SOS/EOS
    """
    data = tokenise(data)
    data = [add_padding(line, context_size) for line in data]
    return data


def unravel_data(data):
    """
    Unravel data into list of words
    """
    data = [word for line in data for word in line]
    return data


class WordEmbedddings:
    # https://github.com/RaRe-Technologies/gensim-data
    def __init__(self, download=False, emedding_type="w2v"):
        self.emedding_type = emedding_type
        self.model_path = f"{self.emedding_type}_model"
        self.model_name = (
            "word2vec-google-news-300"
            if self.emedding_type == "w2v"
            else "glove-wiki-gigaword-50"
        )

        self.embeddings = gensim_downloader.load(self.model_name)

        # if download:
        #     self.embeddings = self.download_pretrained()
        # else:
        #     if emedding_type == "w2v":
        #         self.embeddings = KeyedVectors.load_word2vec_format(self.model_path)
        #     # else:
        #     #     self.embeddings = gensim.models.KeyedVectors.load(
        #     #         f"{self.emedding_type}_model.pth"
        #     #     )

        self.embedding_size = EMBEDDING_DIM

        custom_tokens = [LEFT_PAD_SYMBOL, RIGHT_PAD_SYMBOL, UNK_TOKEN]
        self.custom_embeddings = {
            token: np.random.rand(self.embedding_size) for token in custom_tokens
        }

    def download_pretrained(self):
        model = gensim_downloader.load(self.model_name)
        # if self.emedding_type == "glove":
        #     glove2word2vec("pretrained_model", "pretrained_model")

        # model = KeyedVectors(model)
        model.save_word2vec_format(self.model_path)
        return model

    def get_word_embedding(self, word):
        """
        Get embedding for word
        """
        try:
            return self.embeddings[word]
        except KeyError:
            if is_padding(word):
                return self.custom_embeddings[word]

            return self.custom_embeddings[UNK_TOKEN]

    def get_embeddings(self, words):
        """
        Get embeddings for list of words
        """
        return [self.get_word_embedding(word) for word in words]


class Corpus(Dataset):
    def __init__(self, context_size=CONTEXT_SIZE, batch_size=BATCH_SIZE, dummy=False):
        self.data_folder = "./processed_data/" if not dummy else "./dummy/"
        self.context_size = context_size
        self.batch_size = batch_size

        (self.train_words, self.dev_words, self.test_words,) = (
            self.load_dummy() if dummy else self.load_all_datasets()
        )

        # self.vocab = list(make_vocab(self.train_words))
        # self.vocab_size = len(self.vocab)
        self.vocab_lookup = set(self.train_words)
        self.vocab = list(self.vocab_lookup)
        self.word_to_index = {word: index for index, word in enumerate(self.vocab)}
        self.word_vectors = WordEmbedddings().get_embeddings(self.vocab)

    def load_dataset(self, dataset_type="train"):
        """
        Load data from file
        """

        data = read_data(f"{self.data_folder}clean_yelp-subset.{dataset_type}.txt")
        data = tokenise_and_pad_text(data, self.context_size)

        if dataset_type == "train":
            data = unravel_data(data)
            data = self.replace_with_unk(data)

        return data

    def load_all_datasets(self):
        return (
            self.load_dataset("train"),
            self.load_dataset("dev"),
            # self.load_dataset("test"),
            "a",
        )

    def load_dummy(self):
        return (
            self.load_dataset("train"),
            self.load_dataset("dev"),
            self.load_dataset("test"),
        )

    def replace_with_unk(self, words):
        # words is a list of words
        vocab = make_vocab(words)

        words = [x if vocab.get(x, 0) > MAX_UNK_FREQ else UNK_TOKEN for x in words]
        return words

    def get_word_onehot(self, word):
        """
        Get onehot representation of word
        """
        index = self.word_to_index[word]
        onehot = np.zeros(len(self.vocab))
        onehot[index] = 1
        return onehot

    def get_word_index(self, word):
        if word not in self.vocab_lookup:
            word = UNK_TOKEN
        return self.word_to_index[word]

    def get_word_vectors(self, words):
        return np.mean(
            np.array([self.word_vectors[self.get_word_index(w)] for w in words]), axis=0
        )

    def __len__(self):
        return len(self.train_words) - self.context_size

    def __getitem__(self, index):
        # ret = (context, word)

        ret = (
            torch.tensor(
                self.get_word_vectors(
                    self.train_words[index : index + self.context_size]
                )
            ),
            torch.tensor(
                self.get_word_onehot(self.train_words[index + self.context_size])
            ),
        )

        # if TO_GPU:
        #     ret.to(DEVICE)

        return ret


class BiLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=EMBEDDING_DIM,
        context_size=CONTEXT_SIZE,
        batch_size=BATCH_SIZE,
        dropout=0.1,
    ):
        super(BiLM, self).__init__()
        self.reshape = nn.Linear(embedding_dim, HIDDEN_LAYER_SIZE)
        self.forward_lstm_1 = nn.LSTM(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
        self.backward_lstm_1 = nn.LSTM(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(HIDDEN_LAYER_SIZE, PROJECTION_SIZE, bias=False)
        self.forward_lstm_2 = nn.LSTM(PROJECTION_SIZE, HIDDEN_LAYER_SIZE)
        self.backward_lstm_2 = nn.LSTM(PROJECTION_SIZE, HIDDEN_LAYER_SIZE)

        self.batch_size = batch_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim

        self.lm_layer = nn.Sequential(
            nn.Linear(HIDDEN_LAYER_SIZE * 2, vocab_size),
            nn.Softmax(),
        )

    def forward(self, x):
        ## input shape:
        # seq_len, batch_size, 128

        # logging.info(f"input shape: {x.shape}")
        # 12:27:33 AM input shape: torch.Size([16, 300])

        # Reshape input
        x = self.reshape(x)

        # LSTM layer 1
        x_fwd = x
        x_backwd = x.flip(dims=[0])
        ## forward feed
        out_fwd_1, (h_fwd_1, _) = self.forward_lstm_1(x_fwd)
        ## backward feed
        out_backwd_1, (h_backwd_1, _) = self.backward_lstm_1(x_backwd)
        # concat across batches
        h1 = torch.stack((h_fwd_1, h_backwd_1)).squeeze(dim=1)

        # # log shape of all tensors
        # logging.info(f"Shapes after LSTM 1:")
        # logging.info(f"input shape: {x.shape}")
        # logging.info(f"out_fwd_1 shape: {out_fwd_1.shape}")
        # logging.info(f"h_fwd_1 shape: {h_fwd_1.shape}")
        # 12:27:33 AM Shapes after LSTM 1:
        # 12:27:33 AM input shape: torch.Size([16, 128])
        # 12:27:33 AM out_fwd_1 shape: torch.Size([16, 128])
        # 12:27:33 AM h_fwd_1 shape: torch.Size([1, 128])

        # dropout
        out_fwd_1 = self.dropout(out_fwd_1)
        out_backwd_1 = self.dropout(out_backwd_1)

        # Projection of main and skip connection
        x_fwd_2 = self.proj(out_fwd_1 + x_fwd)
        x_backwd_2 = self.proj(out_backwd_1 + x_backwd)

        # LSTM layer 2
        out_fwd_2, (h_fwd_2, _) = self.forward_lstm_2(x_fwd_2)
        out_backwd_2, (h_backwd_2, _) = self.backward_lstm_2(x_backwd_2)
        h2 = torch.stack((h_fwd_2, h_backwd_2)).squeeze(dim=1)

        # vec = torch.stack((out_fwd_2, out_backwd_2)).squeeze(dim=0)
        vec = torch.cat((out_fwd_2, out_backwd_2), dim=1)
        pred = self.lm_layer(vec)

        # out_fwd = self.lm_layer_fwd(out_fwd_2)
        # out_backwd = self.lm_layer_backwd(out_backwd_2)

        return pred, h1, h2


def train_lm(model, dataset, num_epochs=1):
    """
    Return trained model and avg losses
    """
    logging.info("Training....")

    # min_pp = np.inf
    # best_model = 0

    dataloader = DataLoader(dataset, batch_size=model.batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch_losses = []

    # dims = set()

    for epoch in range(num_epochs):
        logging.info(f"EPOCH: {epoch}")
        model.to(DEVICE)
        model.train()
        losses = []

        for _, (X, y) in enumerate(dataloader):
            if TO_GPU:
                X = X.to(DEVICE)
                y = y.to(DEVICE)
            X = X.float()
            # y = y.long()

            # Prediction
            pred, _, _ = model(X)
            loss = criterion(pred, y)

            # dims.add((X.shape, y.shape, pred.shape))

            # Back propagation
            optimizer.zero_grad()
            loss.backward()

            # GD step
            optimizer.step()
            losses.append(loss.item())

        torch.save(model, f"{MODEL_DIR}model_{epoch}.pth")

        # _, pp = get_text_perplexity(
        #     text=dataset.dev_words,
        #     model=model,
        #     dataset=dataset,
        # )

        loss = np.mean(losses)

        epoch_losses.append(loss)

        with open("losses.txt", "a") as f:
            f.write(f"{epoch}\t{loss}\n")

    #     logging.info(f"{pp}")

    #     if pp < min_pp:
    #         min_pp = pp
    #         best_model = epoch

    # logging.info(f"Best model: {best_model}")
    # logging.info(f"Min perplexity: {min_pp}")

    # print(f"Dimensions: ", dims)

    # model = torch.load(f"./model_{best_model}.pth")

    return model, epoch_losses


class ElMO(nn.Module):
    def __init__(
        self,
        bilm: BiLM,
        # embedding_dim=EMBEDDING_DIM,
        # context_size=CONTEXT_SIZE,
        # batch_size=BATCH_SIZE,
        dropout=0.1,
    ):
        super(ElMO, self).__init__()

        self.bilm = bilm
        self.mixture_size = 2 * 1

        self.scalar_parameters = ParameterList(
            [
                Parameter(
                    torch.FloatTensor([0.0] * self.mixture_size),
                    requires_grad=True,
                )
            ]
        )
        self.gamma = Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        print("In ELMO forward")
        print(f"Input: {inputs.shape}")
        with torch.no_grad():
            _, h1, h2 = self.bilm(inputs)
        hidden_layers = [h1, h2]
        print(f"h1: {h1.shape}")

        normed_weights = torch.nn.functional.softmax(
            torch.cat([parameter for parameter in self.scalar_parameters]), dim=0
        )
        print(f"Weights: {normed_weights.shape}")

        normed_weights = torch.split(normed_weights, split_size_or_sections=1)
        # print(f"Weights split: {normed_weights.shape}")

        embeddings = sum(
            [weight * tensor for weight, tensor in zip(normed_weights, hidden_layers)]
        )
        print(f"Embeddings: {embeddings.shape}")

        embeddings = self.gamma * embeddings
        print(f"Embeddings*gamma: {embeddings.shape}")

        embeddings = self.dropout(embeddings)

        return embeddings


class CorpusYelp(Dataset):
    def __init__(self, dummy=False, dataset_type="train"):
        self.data_folder = "./anlp-assgn2-data/" if not dummy else "./dummy/"
        self.dataset_type = dataset_type

        self.sent, self.labels = self.load_dataset(dataset_type=self.dataset_type)
        # print(self.sent)

        self.sent = self.replace_with_unk(self.sent)
        # print(self.sent)

        self.vocab = set()
        for sent in self.sent:
            self.vocab.update(sent)
        self.vocab = list(self.vocab)
        self.word_to_index = {word: index for index, word in enumerate(self.vocab)}

        logging.info(f"Getting embeddings...")
        self.word_vectors = WordEmbedddings().get_embeddings(self.vocab)
        logging.info(f"Embeddings loaded!")

        self.sent = [
            [self.word_vectors[self.get_word_index(w)] for w in s] for s in sent
        ]
        self.labels = self.onehot_encode(self.labels)

    def get_word_index(self, word):
        return self.word_to_index.get(word, self.word_to_index[UNK_TOKEN])

    def load_dataset(self, dataset_type="train"):
        """
        Load data from file
        """
        data = pd.read_csv(f"{self.data_folder}yelp-subset.{dataset_type}.csv")
        data = data.fillna("0").astype(str)

        data["text"] = data["text"].apply(lambda x: tokenise_and_pad_text(x, 1))
        if dataset_type == "train":
            data["text"] = data["text"].apply(unravel_data)

        return data["text"].to_list(), data["label"].to_list()

    def onehot_encode(self, inputs):
        unq = list(set(inputs))
        n = len(unq)
        identity = np.identity(n, dtype=int)

        unq = {unq[i]: i for i in range(len(unq))}
        encoded = [identity[unq[x]] for x in inputs]

        return encoded

    def replace_with_unk(self, sentences, max_unk_freq=MAX_UNK_FREQ):
        vocab = make_vocab(unravel_data(sentences))
        return [
            [w if vocab.get(w, 0) > max_unk_freq else UNK_TOKEN for w in s]
            for s in sentences
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sentence = self.sent[index]
        if len(sentence) > MAX_SENT_LEN:
            sentence = sentence[:MAX_SENT_LEN]
        ret = (
            torch.tensor(np.mean(sentence, axis=0)),
            torch.tensor(self.labels[index]),
        )
        return ret


class Classifier(nn.Module):
    def __init__(self, bilm, num_classes=5) -> None:
        super().__init__()

        self.embeddings = ElMO(bilm)
        self.layer = nn.Sequential(
            nn.Linear(HIDDEN_LAYER_SIZE, num_classes), nn.Softmax(dim=1)
        )

    def forward(self, x):
        print("fwd in Classifier")
        print(x.shape)
        x = self.embeddings(x)
        print(x.shape)

        return self.layer(x)


def train_elmo(model, dataset, num_epochs=1):
    """
    Return trained model and avg losses
    """
    logging.info("Training....")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch_losses = []

    for epoch in range(num_epochs):
        logging.info(f"EPOCH: {epoch}")
        model.to(DEVICE)
        model.train()
        losses = []

        for _, (X, y) in enumerate(dataloader):
            if TO_GPU:
                X = X.to(DEVICE)
                y = y.to(DEVICE)
            X = X.float()
            # y = y.long()

            # Prediction
            pred = model(X)

            print("Before loss")
            print(pred.shape, X.shape, y.shape)

            loss = criterion(pred, y)

            # dims.add((X.shape, y.shape, pred.shape))

            # Back propagation
            optimizer.zero_grad()
            loss.backward()

            # GD step
            optimizer.step()
            losses.append(loss.item())

        torch.save(model, f"{MODEL_DIR}elmo_model_{epoch}.pth")

        loss = np.mean(losses)
        epoch_losses.append(loss)

        with open("elmo_losses.txt", "a") as f:
            f.write(f"{epoch}\t{loss}\n")

    return model, epoch_losses


if __name__ == "__main__":
    logging.info("Loading Corpus....")

    corpus = CorpusYelp(dummy=True)
    bilm = torch.load(f"{MODEL_DIR}model_0.pth")
    model = Classifier(bilm)
    model, losses = train_elmo(model, corpus, num_epochs=2)
    logging.info("Losses")
    print(losses)

    # Write losses of all epochs to file
    with open("all_elmo_losses.txt", "w") as f:
        for i, loss in enumerate(losses):
            f.write(f"{i}\t{loss}\n")
    x = input().strip()

    if x == "train_bilm":
        logging.info("Loading Corpus....")
        corpus = Corpus(dummy=False)
        logging.info("Corpus loaded")

        model = BiLM(vocab_size=len(corpus.vocab))
        model, losses = train_lm(model, corpus, num_epochs=5)

        logging.info("Losses")
        print(losses)

        # Write losses of all epochs to file
        with open("all_losses.txt", "w") as f:
            for i, loss in enumerate(losses):
                f.write(f"{i}\t{loss}\n")
