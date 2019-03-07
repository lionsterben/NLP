import sys
sys.path.append("/home/FuDawei/NLP/Machine_Translation/data_util")
from nltk.translate.bleu_score import sentence_bleu
from embedding import get_embedding, get_wordset
from collections import defaultdict
import ujson as json
from nltk.tokenize import word_tokenize

_PAD = "<pad>"
_EOS = "<eos>" # end of a sentence
_SOS = "<sos>" # start of input in decoder
_UNK = "<unk>"
_START_VOCAB = [_PAD, _UNK, _EOS, _SOS]
PAD_ID = 0
UNK_ID = 1
EOS_ID = 2
SOS_ID = 3

def eval(preds, truths):
    """pred:list of sentences, sentence is list of words"""
    res_score = 0
    for idx in range(len(preds)):
        pred, truth = preds[idx], truths[idx]
        score = sentence_bleu([pred], truth)
        res_score += score
    return res_score/len(preds)





def data_from_json(filename):
    """Loads JSON data from filename and returns"""
    with open(filename) as data_file:
        data = json.load(data_file)
    return data

def dump_data(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f)

def get_data(source_path, target_path, source_embedding_path, target_embedding_path):
    # source_word, target_word = set(), set()
    ## filter word need more count
    source_word_dict, target_word_dict = defaultdict(int), defaultdict(int)
    # source_path = "/home/FuDawei/NLP/Machine_Translation/dataset/datum2017/Book1_en.txt"
    # target_path = "/home/FuDawei/NLP/Machine_Translation/dataset/datum2017/Book1_cn.txt"
    get_wordset(source_path, source_word_dict)
    # get_wordset(source_dev_path, source_word_dict)
    # get_wordset(target_train_path, target_word_dict)
    get_wordset(target_path, target_word_dict)

    # source_thresh_cnt = 1
    # target_thresh_cnt = 8
    # source_word = list(filter(lambda x: source_word_dict[x]>source_thresh_cnt, source_word_dict.keys()))
    # target_word = list(filter(lambda x: target_word_dict[x]>target_thresh_cnt, target_word_dict.keys()))
    source_word = list(source_word_dict.keys())
    target_word = list(target_word_dict.keys())

    save_dir = "/home/FuDawei/NLP/Machine_Translation/dataset/"

    # path, Word, glove_dim, ignore_head, debug=True
    target_embedding, target_word2id, target_id2word = get_embedding(target_embedding_path, target_word, 300, 1)
    print("target "+str(len(target_word2id)))
    source_embedding, source_word2id, source_id2word = get_embedding(source_embedding_path, source_word, 300, 0)
    print("source "+str(len(source_word2id)))
    
    dump_data(source_embedding, save_dir+"source_embedding.json")
    dump_data(source_word2id, save_dir+"source_word2id.json")
    dump_data(source_id2word, save_dir+"source_id2word.json")

    dump_data(target_embedding, save_dir+"target_embedding.json")
    dump_data(target_word2id, save_dir+"target_word2id.json")
    dump_data(target_id2word, save_dir+"target_id2word.json")

def tokenize_file(file_path):
    lines = [word_tokenize(line.lower()) for line in open(file_path, "r")]
    return lines

def get_origin_data(file_path, word2id_path, id2word_path):
    # lines = [word_tokenize(line.lower()) for line in open(file_path, "r")]
    with open(file_path, "r") as f:
        lines = json.load(f)
    with open(word2id_path, "r") as f:
        word2id = json.load(f)
    with open(id2word_path, "r") as f:
        id2word = json.load(f)
    return lines, word2id, id2word

def generate_batch(lines, word2id, id2word, batch_size, start_index, add_start, add_end, max_len):
    """
    add_start : add SOS token
    add_end   : add EOS token
    input_lines : for decoder input
    output_lines : for decoder output
    """
    batch_lines = lines[start_index: start_index+batch_size]
    if add_start:
        batch_lines = [["<sos>"] + line for line in batch_lines]
    if add_end:
        batch_lines = [line + ["<eos>"] for line in batch_lines]
    batch_lines = [line[:max_len] for line in batch_lines]
    batch_len = [len(line) for line in batch_lines]
    batch_max_len = max(batch_len)

    input_token = [[w for w in line[:-1]] for line in batch_lines]
    output_token = [[w for w in line[1:]] for line in batch_lines]

    input_lines = [[word2id[w] if w in word2id else word2id["<unk>"] for w in line[:-1]]+[0]*(batch_max_len-len(line)) for line in batch_lines]
    output_lines = [[word2id[w] if w in word2id else word2id["<unk>"] for w in line[1:]]+[0]*(batch_max_len-len(line)) for line in batch_lines]
    return input_lines, output_lines, input_token, output_token

    
    




if __name__ == "__main__":
    # base_dir = "/home/FuDawei/NLP/Machine_Translation/dataset/debug/"
    # get_data(base_dir+"endebug", base_dir+"frdebug", "/home/FuDawei/NLP/Embedding/English/glove.840B.300d.txt", "/home/FuDawei/NLP/Embedding/French/cc.fr.300.vec")
    save_dir = "/home/FuDawei/NLP/Machine_Translation/dataset/en-fr-no/"
    base_dir = "/home/FuDawei/NLP/Machine_Translation/dataset/"
    # entrain_lines = tokenize_file(base_dir+"entrain")
    # endev_lines = tokenize_file(base_dir+"endev")
    # frtrain_lines = tokenize_file(base_dir+"frtrain")
    frdev_lines = tokenize_file(base_dir+"frdev")

    # dump_data(entrain_lines, save_dir+"entrain_lines")
    # dump_data(endev_lines, save_dir+"endev_lines")
    # dump_data(frtrain_lines, save_dir+"frtrain_lines")
    dump_data(frdev_lines, save_dir+"frdev_lines")