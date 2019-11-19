# encoding: utf-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from gensim.models import FastText, Word2Vec
import pandas as pd
import numpy as np
import optparse


class _BatchWordEmbeddings(nn.Module):
    def __init__(self, word2ind, emb_dim=300, freeze=True, pad='<PAD>', unk_tok='<UNK>',
                 sos='<SOS>', eos='<EOS>', pretrain_pth=None, saveModel=None, use_cuda=False):
        super(_BatchWordEmbeddings, self).__init__()
        word2indexacal = {k: i for i, k in enumerate(word2ind.keys())}
        word2indexacal[unk_tok] = len(word2indexacal)
        word2indexacal[pad] = len(word2indexacal)
        embeddings = nn.Embedding(len(word2indexacal), emb_dim, padding_idx=word2indexacal[pad])
        embeddings.weight.data.normal_(0.0, 1.0)
        embed_info = {'matrix': embeddings, 'word2indx': word2indexacal}

        self.embeddings = embed_info['matrix']
        self.word2indx = embed_info['word2indx']

        self.freeze = freeze
        self.pad = pad
        self.unk_tok = unk_tok
        self.sos = sos
        self.eos = eos
        self.saveModel = saveModel
        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

    def sentences2Indexes(self, sentences, labels, pad_sent=False, batch_first=False):
        # TODO this is redundant calculations to call this separately
        sequences = [[None]] * len(sentences)
        for i, s in enumerate(sentences):
            indices = [self.word2indx[t] if t in self.word2indx else self.word2indx['<UNK>'] for t in s.split()]
            # indices = [self.word2indx[self.sos]] + indices + [self.word2indx[self.eos]]
            sequences[i] = [torch.tensor(indices).long(), len(indices), labels[i]]
        sorted_seq = sorted(sequences, key=lambda x: x[1], reverse=True)
        sequences = [ent[0] for ent in sorted_seq]
        sorted_labels = [ent[2] for ent in sorted_seq]
        if pad_sent:
            sequences = pad_sequence(sequences, batch_first=batch_first, padding_value=self.word2indx['<PAD>'])
            if self.use_cuda:
                sequences = sequences.cuda()
            return sequences

        return sorted_seq, sorted_labels

    def forward(self, sentences, labels, batch_first=False, pack_seq=False):
        # assumes sentences is a list of lists
        # each entry of each list is a word we want an embedding of

        # if list of strings, we split into words here else just use the provided list of indices
        if isinstance(sentences[0], str):
            sorted_seq, sorted_labels = self.sentences2Indexes(sentences, labels, pad_sent=False,
                                                               batch_first=batch_first)
        else:
            sequences = sentences

        sequences = [s[0] for s in sorted_seq]
        lengths = [s[1] for s in sorted_seq]

        padded = pad_sequence(sequences, batch_first=batch_first, padding_value=self.word2indx['<PAD>'])

        # lengths = torch.LongTensor(lengths)
        if self.use_cuda:
            padded = padded.cuda()
            # lengths = lengths.cuda()

        embeddings = self.embeddings(padded)
        if pack_seq:
            embeddings = pack_padded_sequence(embeddings, lengths, batch_first=batch_first)
        if self.use_cuda:
            embeddings = embeddings.cuda()
        return embeddings, sorted_labels

    def loadFastText(self, model_pth, vocab):
        if '.bin' in model_pth:
            model = FastText.load_fasttext_format(model_pth)
        else:
            model = FastText.load(model_pth)
        self.create_embeddings(model, vocab)

    def loadLRIC(self, model_pth, vocab):
        if '.mdl' in model_pth:
            model = Word2Vec.load(model_pth)
        self.create_embeddings(model, vocab)

    def create_embeddings(self, model, vocab):
        embeddings = []
        word2indx = {}
        for i, word in enumerate(vocab):
            word = word.strip()
            try:
                embeddings.append(torch.from_numpy(model.wv[word]).float())
                word2indx[word] = i
            except:
                embeddings.append(torch.randn(model.wv.vector_size))
                word2indx[word] = i
        embeddings.append(torch.zeros(model.wv.vector_size))
        word2indx[self.pad] = len(word2indx)
        embeddings.append(torch.rand(model.wv.vector_size))
        word2indx[self.unk_tok] = len(word2indx)

        embeddings = torch.stack(embeddings)
        embed_info = {'matrix': embeddings, 'word2indx': word2indx}
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=self.freeze)
        self.word2indx = word2indx

        if self.saveModel is not None:
            torch.save(embed_info, self.saveModel)

    def getVocab(self):
        return self.word2indx.keys()


class _GRU_dense(nn.Module):
    def __init__(self, hidden_dim, n_lstm_layers, emb_dim, dropout_p, classes, word2indx, use_cuda):
        super(_GRU_dense, self).__init__()
        self.embeddings = _BatchWordEmbeddings(word2indx, use_cuda=use_cuda)
        self.encoder = nn.GRU(emb_dim, hidden_dim, num_layers=n_lstm_layers, bidirectional=False, dropout=0)
        self.dense = nn.Linear(hidden_dim, 1000)
        self.dropout = nn.Dropout(dropout_p)
        self.dense2 = nn.Linear(1000, 1000)
        self.predictor = nn.Linear(1000, classes)

    def forward(self, seq, labels):
        seq, labels = self.embeddings(seq, labels, batch_first=False)

        output, hidden = self.encoder(seq)
        hidden = hidden.squeeze(0)

        output = self.dense(hidden)
        output = F.relu(output)
        output = self.dropout(output)

        output = self.dense2(output)
        output = F.relu(output)

        output = self.dense2(output)
        output = F.relu(output)

        preds = self.predictor(output)
        return preds, labels


class EmotionNet():
    '''
    EmotionNet
    '''

    def __init__(self):
        self.__use_cuda = torch.cuda.is_available()
        self.__device = torch.device("cuda" if self.__use_cuda else "cpu")
        self.n___gpu = torch.cuda.device_count()

        self.__max_seq_length = 50
        self.__batch_size = 64

        embed_size = 300
        nh = 30
        drop = 0.5
        n_lstm_layers = 1

        self.__word2indx = torch.load(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.join('resources', 'word2ind.pt')))

        self.__label2indx = torch.load(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.join('resources', 'lab2ind.pt')))
        self.__indx2label = {v: k for k, v in self.__label2indx.items()}
        classes = len(self.__label2indx)

        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.join('resources', 'model.pt'))
        self.__join_model_files(model_path)

        self.__model = _GRU_dense(nh, n_lstm_layers, embed_size, drop, classes, self.__word2indx, self.__use_cuda)
        if self.__use_cuda:
            model = self.__model.cuda()
            model.load_state_dict(torch.load(model_path)['state_dict'])
        else:
            self.__model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['state_dict'])

        self.__model.eval()

    def __join_model_files(self, model_file):
        if os.path.exists(model_file):
            return
        large_file = open(model_file, 'wb')
        chunk_id = 1

        while os.path.exists(model_file + '-%s' % chunk_id):
            with open(model_file + '-%s' % chunk_id, 'rb') as chunk:
                content = chunk.read()
                while content:
                    large_file.write(content)
                    large_file.flush()
                    content = chunk.read()
            chunk_id += 1
        large_file.close()

    def predict(self, text=None, path=None, language='ar', with_dist=False):
        list_scores = []
        if text is not None:
            x = self.__simple_tokenizer([text])
            y = [0]
            y = torch.LongTensor(y)
            if self.__use_cuda:
                y = y.cuda()
            preds, y = self.__model(x, y)
            list_scores.append(preds.cpu().detach().numpy())
        elif path is not None:
            if not os.path.exists(path):
                raise Exception("File not found %s" % path)
            test_file = open(path, 'r', encoding='utf-8')
            test_iterator = pd.read_csv(test_file, sep='\t', chunksize=self.__batch_size)

            with torch.no_grad():
                for batch in test_iterator:
                    x = self.__simple_tokenizer(batch['content'].astype(str))
                    y = [0] * len(batch)
                    y = torch.LongTensor(y)
                    if self.__use_cuda:
                        y = y.cuda()
                    preds, y = self.__model(x, y)
                    list_scores.append(preds.cpu().detach().numpy())

        preds = np.concatenate(list_scores, axis=0)
        preds = torch.from_numpy(preds)
        preds = F.softmax(preds, dim=1)
        values, predicted_class = torch.topk(preds, k=1, dim=1)
        values = values.numpy().flatten()
        labels = [self.__indx2label[indx[0]] for indx in predicted_class.numpy()]
        dist = [dict(zip(self.__label2indx.keys(), p)) for p in preds.numpy()]

        return list(zip(labels, values, dist)) if with_dist else list(zip(labels, values))

    def __simple_tokenizer(self, str):
        sample = [self.__tweet_clean(y) for y in str]
        return sample

    def __tweet_clean(self, text):
        seq = text.split(' ')
        seq = seq[0:self.__max_seq_length]
        if len(seq) == 0:
            seq.append('<UNK>')
        text = ' '.join(seq)
        return text

    def __label_clean(self, str, label2ind):
        sample = [int(label2ind[y]) for y in str]
        return sample


def main():
    parser = optparse.OptionParser()
    parser.add_option('-b', '--batch', action="store", default=None,
                      help='specify a file path on the command line')
    parser.add_option('-d', '--dist', action='store_true', default=False, help='show full distribution over languages')

    options, args = parser.parse_args()

    identifier = EmotionNet()
    if options.batch is not None:
        # "==== Batch Mode ===="
        predictions = identifier.predict(text=None, path=options.batch, with_dist=options.dist)
        print(predictions)
    else:
        import sys
        if sys.stdin.isatty():
            # "==== Interactive Mode ===="
            while True:
                try:
                    print(">>>", end=' ')
                    text = input()
                except Exception as e:
                    print(e)
                    break
                predictions = identifier.predict(text=text, path=None, with_dist=options.dist)
                print(predictions)
        else:
            # "==== Redirected Mode ===="
            lines = sys.stdin.read()
            predictions = []
            for text in lines:
                predictions.extend(identifier.predict(text=text, path=None, with_dist=options.dist))
            print(predictions)


if __name__ == "__main__":
    # main()
    em = EmotionNet()
    p = em.predict(text='Sat in a gym surrounded by big , sweaty Yorkshire men !')
    # p = em.predict(path='C:/Users/hazadeh/WorkStations/PycharmProjects/emotion/emotion/data/test.tsv', with_dist=True)
    for line in p:
        print(line)
