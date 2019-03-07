from data_util.embedding import PAD_ID, EOS_ID, SOS_ID, UNK_ID
import random
import re
import numpy as np
from nltk.tokenize import word_tokenize
# 获取context length，除去padding
class Batch():
    def __init__(self, source_context, target_context, source_length, target_length, source_id, target_id, source_maxlen, target_maxlen):
        ## shape (batch_size, seq_len)
        ## cn_length/en_length (batch_size)
        self.source_context = source_context
        self.source_id = source_id
        self.source_length = source_length
        self.source_maxlen = source_maxlen
        self.target_context = target_context
        self.target_id = target_id
        self.target_length = target_length
        self.target_maxlen = target_maxlen

# def removeStart(sentence):
#     return sentence.lstrip("1234567890. ")

# def split_by_whitespace(sentence):
#     sentence = sentence.lower()
#     words = []
#     for space_separated_fragment in sentence.strip().split():
#         words.extend(re.split(" ", space_separated_fragment))
#     return [w for w in words if w]

def padded(token_batch, batch_pad=0):
    """
    Inputs:
      token_batch: List (length batch size) of lists of ints.
      batch_pad: Int. Length to pad to. If 0, pad to maximum length sequence in token_batch.
    Returns:
      List (length batch_size) of padded of lists of ints.
        All are same length - batch_pad if batch_pad!=0, otherwise the maximum length in token_batch
    """
    maxlen = max(map(lambda x: len(x), token_batch)) if batch_pad == 0 else batch_pad
    return list(map(lambda token_list: token_list + [PAD_ID] * (maxlen - len(token_list)), token_batch)), maxlen

def sentence2id(word2id, sentence):
    # sentence = removeStart(sentence)
    sentence = sentence.lower()
    tokens = word_tokenize(sentence) + ["<eos>"]
    ids = [word2id.get(w, UNK_ID) for w in tokens] + [EOS_ID]
    return tokens, ids

def refill_batches(batches, source_file, source_word2id, target_file, target_word2id, batch_size):
    examples = []
    source_line, target_line = source_file.readline(), target_file.readline()
    source_line, target_line = source_line.lower(), target_line.lower()
    while source_line and target_line:
        source_context, source_id = sentence2id(source_word2id, source_line)
        target_context, target_id = sentence2id(target_word2id, target_line)
        source_len, target_len = len(source_context), len(target_context)
        source_line, target_line = source_file.readline(), target_file.readline()
        examples.append((source_context, source_id, source_len, target_context, target_id, target_len))

        if len(examples) == 160 * batch_size:
            break
    
    for idx in range(0, len(examples), batch_size):
        batch = examples[idx:idx+batch_size]
        #按照source length降序排列
        batch = sorted(batch, key=lambda x: x[2], reverse=True)
        source_context, source_id, source_length, target_context, target_id, target_length = zip(*batch)
        batches.append((source_context, source_id, source_length, target_context, target_id, target_length))
    random.shuffle(batches)
    return 

        


def get_batch_generator(source_word2id, target_word2id, source_path, target_path, batch_size):
    source_file, target_file = open(source_path), open(target_path)
    batches = []
    while True:
        if len(batches) == 0:
            refill_batches(batches, source_file, source_word2id, target_file, target_word2id, batch_size)
        if len(batches) == 0:
            break
        source_context, source_id, source_length, target_context, target_id, target_length = batches.pop(0)

        ## pad source and target id
        source_id, source_maxlen = padded(source_id, 0)
        target_id, target_maxlen = padded(target_id, 0)

        ## transform int into np array
        source_id = np.array(source_id)
        target_id = np.array(target_id)
        source_length = np.array(source_length)
        target_length = np.array(target_length)

        batch = Batch(source_context, target_context, source_length, target_length, source_id, target_id, source_maxlen, target_maxlen)
        yield batch
    return 

        
        


