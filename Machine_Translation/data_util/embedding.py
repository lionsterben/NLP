import numpy as np
import re
import ujson as json
from collections import defaultdict
from nltk.tokenize import word_tokenize
## now we donnot use pretrained embedding
_PAD = "<pad>"
_EOS = "<eos>" # end of a sentence
_SOS = "<sos>" # start of input in decoder
_UNK = "<unk>"
_START_VOCAB = [_PAD, _UNK, _EOS, _SOS]
PAD_ID = 0
UNK_ID = 1
EOS_ID = 2
SOS_ID = 3


def get_wordset(file_path, Word):
    ##word = set()
    with open(file_path, "r") as f:
        data = f.read()
        lines = data.split("\n")
        for line in lines:
            line = line.lower()
            for i in word_tokenize(line):
                # Word.add(i) 
                Word[i] += 1

def get_wordlist(file_path, length):
    ##word = set()
    Word = defaultdict(int)
    with open(file_path, "r") as f:
        data = f.read()
        lines = data.split("\n")
        for line in lines:
            line = line.lower()
            for word in word_tokenize(line):
                Word[word] += 1
    # word_list = Word.most_common(length)
    sort_Word = sorted(Word.items(), key=lambda x:x[1], reverse=True)
    word_list = []
    cnt = 0
    while cnt < length:
        word_list.append(sort_Word[cnt][0])
        cnt += 1
    return word_list

def build_word2id(word_list):
    word2id, id2word = {}, {}
    for idx, key in enumerate(_START_VOCAB):
        word2id[key] = idx
        id2word[idx] = key
    for idx, key in enumerate(word_list):
        word2id[key] = idx + 4
        id2word[idx+4] = key
    return word2id, id2word

def main():
    save_dir = "/home/FuDawei/NLP/Machine_Translation/dataset/en-fr-no/"
    fr_word_list = get_wordlist("/home/FuDawei/NLP/Machine_Translation/dataset/europarl-v7.fr-en.fr", 50000)
    en_word_list = get_wordlist("/home/FuDawei/NLP/Machine_Translation/dataset/europarl-v7.fr-en.en", 50000)
    fr_word2id, fr_id2word = build_word2id(fr_word_list)
    en_word2id, en_id2word = build_word2id(en_word_list)

    with open(save_dir+"fr_word_list.json", "w") as f:
        json.dump(fr_word_list, f)
    with open(save_dir+"en_word_list.json", "w") as f:
        json.dump(en_word_list, f)

    with open(save_dir+"fr_word2id.json", "w") as f:
        json.dump(fr_word2id, f)
    with open(save_dir+"fr_id2word.json", "w") as f:
        json.dump(fr_id2word, f)

    with open(save_dir+"en_word2id.json", "w") as f:
        json.dump(en_word2id, f)
    with open(save_dir+"en_id2word.json", "w") as f:
        json.dump(en_id2word, f)

if __name__ == "__main__":
    main()

    


            
            


## Chinese 352192 300 1
## English 4e5+1 300 0
def get_embedding(path, Word, glove_dim, ignore_head):
    print(path+" embedding start")
    # vocab_size = len(Word) + len(_START_VOCAB)
    # embedding_matrix = np.zeros((vocab_size, glove_dim))
    word2id, id2word = {}, {}
    # embedding_matrix[:len(_START_VOCAB)] = np.random.randn(len(_START_VOCAB), glove_dim)
    embedding_dict = {}
    # _START_VOCAB = [_UNK, _EOS, _SOS]

    idx = 0
    for word in _START_VOCAB:
        word2id[word] = idx
        id2word[idx] = word
        idx += 1

    with open(path, "r") as f:
        line = f.readline()
        if ignore_head:
            # print(line)
            line = f.readline()
        # print(line)
        while line:
            # print(line)
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            if word in Word and word not in word2id:
                vector = list(map(float, line[1:]))
                if len(vector) != glove_dim:
                    raise Exception("wrong word vector dimension in %s"%(path))
                embedding_dict[idx] = vector
                word2id[word] = idx
                id2word[idx] = word
                idx += 1
            line = f.readline()
        vocab_size = len(word2id)
        embedding_matrix = np.zeros((vocab_size, glove_dim))
        embedding_matrix[:len(_START_VOCAB)] = 0.15*np.random.randn(len(_START_VOCAB), glove_dim)
        for idx in range(len(_START_VOCAB), vocab_size):
            embedding_matrix[idx, :] = embedding_dict[idx]
    print(path+" embedding is ok")


    return embedding_matrix, word2id, id2word





