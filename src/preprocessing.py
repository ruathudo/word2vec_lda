import numpy as np
from nltk import FreqDist


class DataProcess:
    SUB_THRESHOLD = 1e-5
    NEG_THRESHOLD = 0.75
    neg_sample_tbl = []
    sub_sample_tbl = []

    def __init__(self,
                 corpus: list = (list,),
                 batch_size: int = 20,
                 neg_sample: int = 5,
                 window_size: int = 2,
                 min_freq: int = 5):

        self.corpus = corpus
        self.batch_size = batch_size
        self.neg_sample = neg_sample
        self.window_size = window_size
        self.min_freq = min_freq

    def gen_vocab(self):
        """
        Create vocab dictionary
        """
        # flatten list of tokens in corpus
        tokens = sum(self.corpus, [])

        # calculate freq of words
        freq_dist = FreqDist(tokens)

        vocab, word2idx, idx2word = {}, {}, {}

        index = 0
        for word, freq in freq_dist.items():
            # ignore the word has freq less than min_freq
            if freq < self.min_freq:
                continue
            vocab[word] = freq
            word2idx[word] = index
            idx2word[index] = word
            index += 1

        return vocab, word2idx, idx2word

    def gen_word_pairs(self, word2idx):
        window_size = self.window_size

        idx_pairs = []
        # for each sentence in tokenized corpus
        for sentence in self.corpus:
            indices = [word2idx[w] for w in sentence if w in word2idx]

            # select center word
            for center_word_pos in range(len(indices)):
                # for each window position
                for w in range(-window_size, window_size + 1):
                    context_word_pos = center_word_pos + w
                    # make sure not jump out sentence
                    if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                        continue
                    context_word_idx = indices[context_word_pos]
                    idx_pairs.append((indices[center_word_pos], context_word_idx))

        # convert to numpy for faster calculation
        idx_pairs = np.array(idx_pairs)

        return idx_pairs

    def gen_negative_sample_table(self, freqs):
        """
        params: freqs is the list of dict {idx: freq} of word
        As mentioned in the blog: http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/

        implemented same as word2vec c code.
        The way this selection is implemented in the C code is interesting. They have a large array with 100M elements
        (which they refer to as the unigram table). They fill this table with the index of each word in the vocabulary
        multiple times, and the number of times a word’s index appears in the table is given by P(wi) * table_size.

            p(w_i) = f(w_i) ** neg_sampling / sum(f(w_i) ** neg_sampling for w_i in vocab)
        """
        # init 100M sample table
        sample_tbl_size = 1e8
        sample_tbl = []

        # calculate the probability for each word by multiply with neg_threshold = 0.75
        pow_freq = np.array(list(freqs.values())) ** self.NEG_THRESHOLD
        # Sum the prob of all words
        pow_total_freq = sum(pow_freq)
        # calculate the ratio for filter
        r = pow_freq / pow_total_freq
        # calculate how many samples each word takes to fill to table
        n_samples = np.round(r * sample_tbl_size)

        # Now fill n_samples (by id) to the table
        for word, count in enumerate(n_samples):
            sample_tbl += [word] * int(count)

        # convert to np array for fast calculation
        self.neg_sample_tbl = np.array(sample_tbl)

        return self.neg_sample_tbl

    def get_neg_sample(self, idx1, idx2):
        """
        get negative sample from table, not equal to input and context word
        """
        negs = []

        while len(negs) < self.neg_sample:
            samples = np.random.choice(self.neg_sample_tbl, size=self.neg_sample - len(negs))
            negs += [i for i in samples if i != idx1 and i != idx2 and i not in negs]

        return negs

    def gen_batches(self, word2idx, idx_pairs):
        """
        generate batches
        from pairs of input and context words + list of negative sampling
        [[u, u, u], [v, v, v], [[neg, neg], [neg, neg], [neg, neg]]]
        """

        batches = []

        for i in range(0, len(idx_pairs), self.batch_size):
            inputs, contexts, neg_samples = [], [], []

            for pair in idx_pairs[i: i + self.batch_size]:
                inputs.append(pair[0])
                contexts.append(pair[1])
                neg_samples.append(self.get_neg_sample(pair[0], pair[1]))

            batches.append([inputs, contexts, neg_samples])

        return batches

    def pipeline(self):
        # generate vocab dictionary
        vocab, word2idx, idx2word = self.gen_vocab()

        # generate input and context words pair
        idx_pairs = self.gen_word_pairs(word2idx)

        # generate negative sample table
        self.gen_negative_sample_table(vocab)

        # create batches
        batches = self.gen_batches(word2idx, idx_pairs)

        return batches, vocab, word2idx, idx2word

