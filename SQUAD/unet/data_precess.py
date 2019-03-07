import ujson as json
import numpy as np
import spacy
import os
import random
from tqdm import tqdm
from collections import Counter
from collections import defaultdict

NULL = "--null--"
OOV = "--oov--"
NA = "noanswer"

para_limit = 450
ques_limit = 50

nlp = spacy.load("en", parser=False)
"""in my experiment, word embedding need fix, other embedding need train"""

def data_from_json(filename):
    """Loads JSON data from filename and returns"""
    with open(filename) as data_file:
        data = json.load(data_file)
    return data

def tokenize(sentence):
    """get sentence token, pos, ner, lemma"""
    doc = nlp(sentence)
    token, tag, ner, lemma = [], [], [], []
    for word in doc:
        token.append(word.text)
        tag.append(word.tag_)
        ner.append(word.ent_type_)
        lemma.append(word.lemma_)
    return token, tag, ner, lemma

def get_char_word_loc_mapping(context, context_tokens):
    """
    Return a mapping that maps from character locations to the corresponding token locations.
    If we're unable to complete the mapping e.g. because of special characters, we return None.

    Inputs:
      context: string (unicode)
      context_tokens: list of strings (unicode)

    Returns:
      mapping: dictionary from ints (character locations) to (token, token_idx) pairs
        Only ints corresponding to non-space character locations are in the keys
        e.g. if context = "hello world" and context_tokens = ["hello", "world"] then
        0,1,2,3,4 are mapped to ("hello", 0) and 6,7,8,9,10 are mapped to ("world", 1)
    """
    acc = '' # accumulator
    current_token_idx = 0 # current word loc
    mapping = dict()

    for char_idx, char in enumerate(context): # step through original characters
        if char != ' ': # if it's not a space:
            acc += char # add to accumulator
            context_token = context_tokens[current_token_idx] # current word token
            if acc == context_token: # if the accumulator now matches the current word token
                syn_start = char_idx - len(acc) + 1 # char loc of the start of this word
                for char_loc in range(syn_start, char_idx+1):
                    mapping[char_loc] = (acc, current_token_idx) # add to mapping
                acc = '' # reset accumulator
                current_token_idx += 1

    if current_token_idx != len(context_tokens):
        return None
    else:
        return mapping

def preprocess(filename, tier, Word, Ner, Pos, ques_counter):
    """trans origin data to examples, every example include context token,  context chars, context pos, context ner, context lemma, 
    context match, context match lemma; question is the same and y1s, y2s(true answer start and end), y1sp, y2sp(plasuible answer start and end)
    id, isimpossible
    filename: train/dev json path, tier: train/dev, Word, Ner, Pos:set"""
    
    examples = []
    drop_in_process = defaultdict(list)
    # Word, Ner, Char, Pos = set(), set(), set(), set()
    dataset = data_from_json(filename)
    cnt = 0
    num_cannot_match = 0
    for articles_id in range(len(dataset['data'])):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]["context"]
            context = ''.join(["noanswer ", context])
            context = context.replace("''", '" ').replace("``", '" ')
            context = context.lower()
            context_token, context_tag, context_ner, context_lemma = tokenize(context)
            # context_chars = [list(word) for word in context_token] 
            context_token_set = set(context_token)
            # context_lemma_set = set(context_lemma)
            context_lemma_set = {lemma if lemma != '-PRON-' else token for lemma, token in zip(context_lemma, context_token)}
            for token in context_token_set:
                Word.add(token)
            for ner in context_ner:
                Ner.add(ner)
            for tag in context_tag:
                Pos.add(tag)
            context_map = get_char_word_loc_mapping(context, context_token)
            if context_map is None:
                # print("context_map cannot match")
                num_cannot_match += len(article_paragraphs[pid]['qas'])
                for qn in article_paragraphs[pid]["qas"]:
                    drop_in_process["id"].append(qn["id"])
                continue
            context_count = Counter(context_token)
            context_len = len(context_token)
            context_tf = [float(context_count[w])/float(context_len) for w in context_token]
            for qn in article_paragraphs[pid]['qas']:
                qn_text = qn["question"].replace("''", '" ').replace("``", '" ')
                qn_text = qn_text.lower()
                qn_token, qn_tag, qn_ner, qn_lemma = tokenize(qn_text)
                for token in qn_token:
                    ques_counter[token] += 1
                # qn_chars = [list(word) for word in qn_token]
                qn_token_set = set(qn_token)
                # qn_lemma_set = set(qn_lemma)
                qn_lemma_set = {lemma if lemma != '-PRON-' else token for lemma, token in zip(qn_lemma, qn_token)}
                # qn_count = Counter(qn_token)
                qn_len = len(qn_token)
                qn_tf = [float(context_count[w])/float(context_len) for w in qn_token]
                for token in qn_token_set:
                    Word.add(token)
                for ner in qn_ner:
                    Ner.add(ner)
                for tag in qn_tag:
                    Pos.add(tag)
                context_match = [w in qn_token_set for w in context_token]
                qn_match = [w in context_token_set for w in qn_token]
                # context_lemma_match, qn_lemma_match = [], []
                # for (lemma, token) in zip(context_lemma, context_token):
                #     cmp = lemma
                #     if cmp == "-PRON-":
                #         cmp = token
                #     context_lemma_match.append(cmp in qn_lemma_set)
                context_lemma_match = [(lemma if lemma != "-PRON-" else token) in qn_lemma_set for (lemma, token) in zip(context_lemma, context_token)]
                # for (lemma, token) in zip(qn_lemma, qn_token):
                #     cmp = lemma
                #     if cmp == "-PRON-":
                #         cmp = token
                #     qn_lemma_match.append(cmp in context_lemma_set)
                qn_lemma_match = [(lemma if lemma != "-PRON-" else token) in context_lemma_set for (lemma, token) in zip(qn_lemma, qn_token)]
                ## because answer only one, if true answer exists, plausible is nonexists

                if qn["answers"]:
                    answer = qn["answers"][0]['text'].lower()
                    answer_start_charloc = qn["answers"][0]['answer_start'] + len("noanswer ")
                    answer_end_charloc = answer_start_charloc + len(answer)
                    if context[answer_start_charloc:answer_end_charloc] != answer:
                        drop_in_process["id"].append(qn["id"])
                        print("answer can not match")
                        continue
                    true_start = context_map[answer_start_charloc][1]
                    true_end = context_map[answer_end_charloc-1][1]
                    assert true_start <= true_end
                    ## check token is same
                    if "".join(context_token[true_start:true_end+1]) != "".join(answer.split()):
                        drop_in_process["id"].append(qn["id"])
                        num_cannot_match += 1
                        continue
                    fake_start, fake_end = true_start, true_end
                elif qn["plausible_answers"]:
                    true_start, true_end = 0, 0
                    plausible_answer = qn["plausible_answers"][0]['text'].lower()
                    fake_answer_start_charloc = qn["plausible_answers"][0]["answer_start"] + len("noanswer ")
                    fake_answer_end_charloc = fake_answer_start_charloc + len(plausible_answer)
                    if context[fake_answer_start_charloc:fake_answer_end_charloc] != plausible_answer:
                        drop_in_process["id"].append(qn["id"])
                        num_cannot_match += 1
                        continue
                    fake_start = context_map[fake_answer_start_charloc][1]
                    fake_end = context_map[fake_answer_end_charloc-1][1]
                    assert fake_start <= fake_end
                    ## check token is same
                    if "".join(context_token[fake_start:fake_end+1]) != "".join(plausible_answer.split()):
                        drop_in_process["id"].append(qn["id"])
                        num_cannot_match += 1
                        continue
                else:
                    print("hola")
                    true_start, true_end, fake_start, fake_end = 0, 0, 0, 0
                is_impossible = qn["is_impossible"]
                example = {"context_token": context_token, "context_pos": context_tag, "context_ner": context_ner,  "context_tf": context_tf, "context_match": context_match, "context_lemma_match": context_lemma_match, 
                "qn_token": qn_token, "qn_pos": qn_tag, "qn_ner": qn_ner,  "qn_tf": qn_tf, "qn_match": qn_match, "qn_lemma_match": qn_lemma_match,
                "true_start": true_start, "true_end": true_end, "fake_start": fake_start, "fake_end": fake_end, "is_impossible": is_impossible, "id": qn["id"], "cnt": cnt}
                cnt += 1
                if cnt % 1000 == 0:
                    print("haha")
                    print(num_cannot_match)
                examples.append(example)
    random.shuffle(examples)
    print("{} can not match".format(num_cannot_match))
    print("there is "+str(len(examples)))
    with open(save_dir + tier+ 'drop_in_process.json', 'w') as f :
        json.dump(drop_in_process, f)
    return examples

## this model is large, so I choose glove 6B 100d 4e5+1
## paper do not say it uses char rnn,, so just word need embedding, ner,pos,lemma need train
def get_embedding(token, emb_file_path, embedd_size):
    ## token: set of word
    embedding_dict = {}
    token2idx = {}
    token2idx[NULL] = 0
    token2idx[OOV] = 1
    token2idx[NA] = 2
    idx = 3
    with open(emb_file_path, 'r') as fh:
        for line in fh:
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            if line[0] in token:
                embedding_dict[word] = list(map(float, line[1:]))
                token2idx[word] = idx
                idx += 1
    vocab_size = len(token2idx) ## include special token
    emb_matrix = np.zeros((vocab_size, embedd_size))
    emb_matrix[:2, :] = np.random.randn(2, embedd_size)
    emb_matrix[2,:] = [np.random.normal(scale=0.01) for _ in range(embedd_size)]
    idx2token = {v:k for k,v in token2idx.items()}
    for idx in range(3, vocab_size):
        token = idx2token[idx]
        emb_matrix[idx,:] = embedding_dict[token]
    return token2idx, emb_matrix

def make_dict(token):
    # OOV = "--oov--"
    """this function is for ner,pos to build token map to idx, special token only need null for padding"""
    token2idx = {NULL: 0, OOV: 1}
    idx = 2
    for tok in token:
        token2idx[tok] = idx
        idx += 1
    return token2idx

## todo batch data and main function 
def build_data(examples, tier, out_file, drop_file, word2idx, pos2id, ner2id):
    """ tier: train/dev """
    count, count_without_drop = 0, 0
    drop = defaultdict(list)

    def filter(example):
        # remove start or end beyond para
        return example['fake_start'] >= para_limit or example['fake_end'] >= para_limit

    context_ids = []
    context_tokens = []
    context_matchs = []
    context_lemma_matchs = []
    context_tfs = []
    context_pos_ids = []
    context_ner_ids = []
    ques_ids = []
    ques_tokens = []
    ques_matchs = []
    ques_lemma_matchs = []
    ques_pos_ids = []
    ques_ner_ids = []
    ques_tfs = []
    true_starts = []
    true_ends = []
    fake_starts = []
    fake_ends = []
    ids = []
    cnts = []
    has_ans = []

    for example in tqdm(examples):
        count += 1
        if filter(example):
            print("one more beyond")
            drop["id"].append(example["id"])
            drop["cnt"].append(example["cnt"])
            continue
        ## question context need pad in head
        count_without_drop += 1
        q_len = len(example["qn_token"])
        pad = 0
        if ques_limit > q_len:
            pad = ques_limit-q_len

        context_idx = np.zeros([para_limit], dtype=np.int32)
        context_match = np.zeros([para_limit], dtype=np.int32)
        context_lemma_match = np.zeros([para_limit], dtype=np.int32)
        context_tf = np.zeros([para_limit], dtype = np.float32)
        context_pos_idx = np.zeros([para_limit], dtype=np.int32)
        context_ner_idx = np.zeros([para_limit], dtype=np.int32)
        ques_idx = np.zeros([ques_limit], dtype=np.int32)
        ques_match = np.zeros([ques_limit], dtype=np.int32)
        ques_lemma_match = np.zeros([ques_limit], dtype=np.int32)
        ques_pos_idx = np.zeros([ques_limit], dtype=np.int32)
        ques_ner_idx = np.zeros([ques_limit], dtype=np.int32)
        ques_tf = np.zeros([ques_limit], dtype=np.float32)

        def _get(dic, key):
            if key in dic:
                return dic[key]
            else:
                return 1
        ## context raw data to Standardized data
        for i, token in enumerate(example["context_token"][:para_limit]):
            context_idx[i] = _get(word2idx, token)
        for i, token in enumerate(example["context_match"][:para_limit]):
            context_match[i] = 1 if token else 0
        for i, token in enumerate(example["context_lemma_match"][:para_limit]):
            context_lemma_match[i] = 1 if token else 0
        for i, token in enumerate(example["context_tf"][:para_limit]):
            context_tf[i] = token
        for i, token in enumerate(example["context_pos"][:para_limit]):
            context_pos_idx[i] = _get(pos2id, token)
        for i, token in enumerate(example["context_ner"][:para_limit]):
            context_ner_idx[i] = _get(ner2id, token)
        
        ## ques raw data to Standardized data
        for i, token in enumerate(example["qn_token"][:ques_limit]):
            i += pad
            ques_idx[i] = _get(word2idx, token)
        for i, token in enumerate(example["qn_match"][:ques_limit]):
            i += pad
            ques_match[i] = 1 if token else 0
        for i, token in enumerate(example["qn_lemma_match"][:ques_limit]):
            i += pad
            ques_lemma_match[i] = 1 if token else 0
        for i, token in enumerate(example["qn_tf"][:ques_limit]):
            i += pad
            ques_tf[i] = token
        for i, token in enumerate(example["qn_pos"][:ques_limit]):
            i += pad
            ques_pos_idx[i] = _get(pos2id, token)
        for i, token in enumerate(example["qn_ner"][:ques_limit]):
            i += pad
            ques_ner_idx[i] = _get(ner2id, token)
        
        true_start, true_end = example["true_start"], example["true_end"]
        fake_start, fake_end = example["fake_start"], example["fake_end"]
        ans = 0 if example["is_impossible"] else 1
        id = example["id"]
        cnt = example["cnt"]

        context_ids.append(context_idx.tolist())
        context_matchs.append(context_match.tolist())
        context_lemma_matchs.append(context_lemma_match.tolist())
        context_tfs.append(context_tf.tolist())
        context_pos_ids.append(context_pos_idx.tolist())
        context_ner_ids.append(context_ner_idx.tolist())

        ques_ids.append(ques_idx.tolist())
        ques_matchs.append(ques_match.tolist())
        ques_lemma_matchs.append(ques_lemma_match.tolist())
        ques_pos_ids.append(ques_pos_idx.tolist())
        ques_ner_ids.append(ques_ner_idx.tolist())
        ques_tfs.append(ques_tf.tolist())
        # print(true_start)
        # print(true_end)
        true_starts.append(true_start)
        true_ends.append(true_end)
        fake_starts.append(fake_start)
        fake_ends.append(fake_end)
        ids.append(id)
        cnts.append(cnt)
        has_ans.append(ans)
        context_tokens.append(example["context_token"][:para_limit])
        qn_token = example['qn_token'][:ques_limit]
        ques_tokens.append(['']*(ques_limit - len(qn_token)) + qn_token)

    print("{}/{} example is valid".format(count_without_drop, count))
    data = {
        "context_ids" : context_ids,
        "context_tokens" : context_tokens,
        "context_matchs" : context_matchs,
        "context_lemma_matchs" : context_lemma_matchs,
        "context_tfs" : context_tfs,
        "context_pos_ids" : context_pos_ids,
        "context_ner_ids" : context_ner_ids,
        "ques_ids" : ques_ids,
        "ques_matchs" : ques_matchs,
        "ques_lemma_matchs" : ques_lemma_matchs,
        "ques_tokens" : ques_tokens,
        "ques_pos_ids" : ques_pos_ids,
        "ques_ner_ids" : ques_ner_ids,
        "ques_tfs": ques_tfs,
        "true_starts" : true_starts,
        "true_ends" : true_ends,
        "fake_starts" : fake_starts,
        "fake_ends" : fake_ends,
        "ids" : ids,
        "cnts" : cnts,
        "total" : count_without_drop,
        "has_ans": has_ans
    }
    
    with open(out_file, "w") as f:
        json.dump(data, f)
    
    with open(drop_file, 'w', encoding='utf-8') as f:
        json.dump(drop, f)

if __name__ == "__main__":
    save_dir = "/home/FuDawei/NLP/SQUAD/unet/data/"
    glove_embedding = "/home/FuDawei/NLP/Embedding/English/glove.6B/glove.6B.100d.txt"
    train_file_path = "/home/FuDawei/NLP/SQUAD/data/train-v2.0.json"
    dev_file_path = "/home/FuDawei/NLP/SQUAD/data/dev-v2.0.json"
    Word, Pos, Ner = set(), set(), set()
    ques_counter = Counter()
    tune_idx = set([0,1,2])
    train_examples = preprocess(train_file_path, "train", Word, Ner, Pos, ques_counter)
    dev_examples = preprocess(dev_file_path, "dev", Word, Ner, Pos, ques_counter)
    word2id, word_emb_matrix = get_embedding(Word, glove_embedding, 100)
    ner2id = make_dict(Ner)
    pos2id = make_dict(Pos)

    for i, (word, _) in enumerate(ques_counter.most_common()):
        if len(tune_idx) > 1000:
            break
        if word in word2id:
            tune_idx.add(word2id[word])


    build_data(train_examples, "train", save_dir+"train.json", save_dir+"train_drop.json", word2id, pos2id, ner2id)
    build_data(dev_examples, "dev", save_dir+"dev.json", save_dir+"dev_drop.json", word2id, pos2id, ner2id)

    with open(save_dir + 'word_emb_matrix.json', 'w', encoding='utf-8') as f :
        json.dump(word_emb_matrix, f)

    with open(save_dir + 'word2id.json', 'w', encoding='utf-8') as f :
        json.dump(word2id, f)

    with open(save_dir + 'pos2id.json', 'w') as f :
        json.dump(pos2id, f)

    with open(save_dir + 'ner2id.json', 'w') as f :
        json.dump(ner2id, f)

    with open(save_dir + "tune_idx.json", "w") as f:
        json.dump(tune_idx, f)



    


    

        
        

        
        
        





    




                    



                
                

                
                
                
                




        