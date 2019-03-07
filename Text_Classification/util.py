import ujson as json
import random
from nltk.tokenize import word_tokenize
from collections import defaultdict

PAD_ID = 0
UNK_ID = 1
pad = "<pad>"
unk = "<unk>"

def build_word2id(word_list):
    word2id = {pad:0, unk:1}
    idx = 2
    for word in word_list:
        word2id[word] = idx
        idx += 1
    return word2id

def get_wordlist(train, dev, length):
    ##word = set()
    Word = defaultdict(int)
    for line, _ in train:
        line = line.lower()
        cc = word_tokenize(line)
        for word in cc:
            Word[word] += 1
    for line, _ in dev:
        line = line.lower()
        cc = word_tokenize(line)
        for word in cc:
            Word[word] += 1
    print(len(Word))
    sort_Word = sorted(Word.items(), key=lambda x:x[1], reverse=True)
    word_list = []
    cnt = 0
    while cnt < length:
        word_list.append(sort_Word[cnt][0])
        cnt += 1
    return word_list
    


def build_data(data, max_len, word2id):
    final_text, final_star = [], []
    for text, star in data:
        token = word_tokenize(text.lower())
        token2id = [word2id.get(word, UNK_ID) for word in token[:max_len]] + [PAD_ID]*(max_len-len(token))
        star = int(star) - 1
        final_text.append(token2id)
        final_star.append(star)
    return (final_text, final_star)

def batch_generator(text, star, batch_size):
    num_exp = len(star)
    for start in range(0, num_exp, batch_size):
        yield (text[start: start+batch_size], star[start: start+batch_size])

def split_data(text, star):
    assert len(text) == len(star)
    example = list(zip(text, star))
    random.shuffle(example)
    train_size = int(len(example)*0.99)
    train, dev = example[:train_size], example[train_size:]
    return train, dev


if __name__ == "__main__":
    # with open("/home/FuDawei/NLP/Text_Classification/dataset/word_list.json", "r") as f:
    #     word_list = json.load(f)
    
    with open("/home/FuDawei/NLP/Text_Classification/dataset/train.json", "r") as f:
        train = json.load(f)
    # clip_text = build_text(text, 30, word2id)
    with open("/home/FuDawei/NLP/Text_Classification/dataset/dev.json", "r") as f:
        dev = json.load(f)
    # train, dev = split_data(clip_text, star)
    # word_list = get_wordlist(train, dev, 30000)




    # word2id = build_word2id(word_list)
    # assert len(word2id) == len(word_list) + 2
    # with open("/home/FuDawei/NLP/Text_Classification/dataset/word2id.json", "w") as f:
    #     json.dump(word2id, f)
    with open("/home/FuDawei/NLP/Text_Classification/dataset/word2id.json", "r") as f:
        word2id = json.load(f)

    train_text, train_star = build_data(train, 100, word2id)
    dev_text, dev_star = build_data(dev, 100, word2id)

    with open("/home/FuDawei/NLP/Text_Classification/dataset/train_text.json", "w") as f:
        json.dump(train_text, f)
    with open("/home/FuDawei/NLP/Text_Classification/dataset/train_star.json", "w") as f:
        json.dump(train_star, f)
    with open("/home/FuDawei/NLP/Text_Classification/dataset/dev_text.json", "w") as f:
        json.dump(dev_text, f)
    with open("/home/FuDawei/NLP/Text_Classification/dataset/dev_star.json", "w") as f:
        json.dump(dev_star, f)



    


    

