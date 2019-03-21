import os
import multiprocessing

import torch
import torch.optim as optim

from src.model import SkipGram
from src.preprocessing import DataProcess
from src.utils import get_data

BASE_DIR = os.path.dirname(os.path.dirname(__file__))


class Word2Vec:
    def __init__(self, lang="english",
                 n_epoch=20,
                 batch_size=500,
                 embed_dim=300,
                 window_size=5,
                 neg_sample=10,
                 min_count=5,
                 lr=0.01,
                 report_every=1):

        self.lang = lang
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.neg_sample = neg_sample
        self.min_count = min_count
        self.lr = lr
        self.report_every = report_every

        self.model, self.optimizer = None, None
        self.batches, self.vocab, self.word2idx, self.idx2word = [], [], [], []

        # check if GPU available
        is_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if is_cuda else "cpu")
        # number of cpu threads for torch
        workers = multiprocessing.cpu_count()
        torch.set_num_threads(workers)

        print("Train session using {}, processor numbers: {}".format(self.device, workers))

    def handle_data(self):
        # get dataset in correct format
        print("Downloading the data")
        train_data, dev_data, test_data = get_data(self.lang)
        # process data for training
        processor = DataProcess(corpus=train_data,
                                batch_size=self.batch_size,
                                neg_sample=self.neg_sample,
                                window_size=self.window_size,
                                min_freq=self.min_count)
        print("Processing data")
        self.batches, self.vocab, self.word2idx, self.idx2word = processor.pipeline()

    def create_model(self):
        print("Initialize model")
        vocab_size = len(self.word2idx)
        self.model = SkipGram(vocab_size=vocab_size, emb_dim=self.embed_dim).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    def train(self):
        print('Start training')
        # print(self.data.gen_batch()[0])
        for epoch in range(self.n_epoch):
            total_loss = 0

            for minibatch in self.batches:
                pos_u = torch.tensor(minibatch[0], dtype=torch.long).to(self.device)
                pos_v = torch.tensor(minibatch[1], dtype=torch.long).to(self.device)
                neg_v = torch.tensor(minibatch[2], dtype=torch.long).to(self.device)

                # print(len(pos_u), len(pos_v), len(neg_v))
                self.optimizer.zero_grad()
                loss = self.model.forward(pos_u, pos_v, neg_v)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if ((epoch + 1) % self.report_every) == 0:
                print('epoch: %d, loss: %.4f' % (epoch + 1, total_loss))

    def save_model(self, filepath):
        print("Saved model in {}".format(filepath))
        self.model.save(filepath, self.idx2word)


if __name__ == '__main__':
    language = input("What language do you want to train? (english, finnish, german...): ")
    epochs = input("How many epochs? (small number for quick test): ")
    # training pipeline
    w2v = Word2Vec(language, n_epoch=int(epochs))
    w2v.handle_data()
    w2v.create_model()
    w2v.train()
    w2v.save_model(os.path.join(BASE_DIR, 'models/w2v.txt'))
