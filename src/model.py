import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):
    """
    Skip-Gram model
    """

    def __init__(self, vocab_size: int, emb_dim: int = 200):
        super(SkipGram, self).__init__()

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

        self.u_embeddings = nn.Embedding(vocab_size, emb_dim, sparse=True)
        self.v_embeddings = nn.Embedding(vocab_size, emb_dim, sparse=True)
        self.init_emb()

    def init_emb(self):
        """
        init the weight as original word2vec
        """
        initrange = 0.5 / self.emb_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(0, 0)

    def forward(self, pos_u, pos_v, neg_v):
        """
        forward process.
        the pos_u and pos_v shall be the same size.
        the neg_v shall be {negative_sampling_count} * size_of_pos_u
        eg:
        5 sample per batch with 300d word embedding and 10 times neg sampling.
        pos_u 5 * 300
        pos_v 5 * 300
        neg_v 5 * 10 * 300

        :param pos_u:  positive pairs u, list
        :param pos_v:  positive pairs v, list
        :param neg_v:  negative pairs v, list
        """
        emb_u = self.u_embeddings(pos_u)  # batch_size * emb_size
        emb_v = self.v_embeddings(pos_v)  # batch_size * emb_size
        emb_neg = self.v_embeddings(neg_v)  # batch_size * neg sample size * emb_size

        pos_score = torch.mul(emb_u, emb_v).squeeze()
        pos_score = torch.sum(pos_score, dim=1)
        pos_score = F.logsigmoid(pos_score)

        neg_score = torch.bmm(emb_neg, emb_u.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-neg_score)

        loss = torch.sum(pos_score) + torch.sum(neg_score)

        return -1 * loss

    def save(self, filepath, idx2word):
        """
        Save the word vectors as standard word2vec format
        """
        embeds = self.u_embeddings.weight.cpu().data.numpy()
        fout = open(filepath, 'w', encoding='utf-8')

        fout.write('%d %d\n' % (len(idx2word), self.emb_dim))

        for idx, word in idx2word.items():
            emb = ' '.join(map(str, embeds[idx]))
            fout.write('%s %s\n' % (word, emb))
