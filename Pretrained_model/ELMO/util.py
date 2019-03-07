import numpy as np
import ujson as json

def _make_bos_eos(
        character: int,
        padding_character: int,
        beginning_of_word_character: int,
        end_of_word_character: int,
        max_word_length: int
):
    char_ids = [padding_character] * max_word_length
    char_ids[0] = beginning_of_word_character
    char_ids[1] = character
    char_ids[2] = end_of_word_character
    return char_ids

def _make_normal_word(word, padding_character, beginning_of_word_character, end_of_word_character, max_word_length):
    char_ids = [padding_character] *  max_word_length
    char_ids[0] = beginning_of_word_character
    idx = 1
    word = word[:max_word_length-2]
    for char in word:
        if ord(char)+1 < 257:
            char_ids[idx] = ord(char) + 1
        else:
            char_ids[idx] = 261
        idx += 1
    char_ids[idx] = end_of_word_character
    return char_ids



def token2id(sentences):
    """
    token: already processed data, list of sentence, sentence has different length
    [["<s>", i", "love", "you", "</s>"], ["<s>", "the", "sky", "is", "blue", "</s>"]]
    padding for sentence is 0, so utf-8 need + 1
    res is batch, max_seq_len, 50


    return: res and word mask
    """
    beginning_of_sentence_character = 257  # <begin sentence>
    end_of_sentence_character = 258  # <end sentence>
    beginning_of_word_character = 259  # <begin word>
    end_of_word_character = 260  # <end word>
    padding_character = 261 # <padding>

    max_seq_len = max([len(sentence) for sentence in sentences])
    batch = len(sentences)
    res = np.zeros((batch, max_seq_len, 50)) # word->char, max_char_len is 50; every sentence has begin and end mask
    mask = np.zeros((batch, max_seq_len))
    for idx, sentence in enumerate(sentences):
        for i in range(len(sentence)):
            mask[idx][i] = 1
    

    beginning_of_sentence_characters = _make_bos_eos(
        beginning_of_sentence_character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        50
    )
    end_of_sentence_characters = _make_bos_eos(
        end_of_sentence_character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        50
    )

    for s_id, sentence in enumerate(sentences):
        for t_id, token in enumerate(sentence):
            if token == "<s>":
                res[s_id, t_id] = beginning_of_sentence_characters
            elif token == "</s>":
                res[s_id, t_id] = end_of_sentence_characters
            else:
                res[s_id, t_id] = _make_normal_word(token, padding_character, beginning_of_word_character, end_of_word_character, 50)
    
    return res, mask

def build_ground(sentences, word2id):
    forward, backward = [], []
    for sentence in sentences:
        forward.append(sentence+["</s>"])
        backward.append(list(reversed(sentence))+["<s>"])
    max_len = max([len(sentence) for sentence in forward])
    batch = len(sentences)
    forward_ground, backward_ground = np.zeros((batch, max_len)), np.zeros((batch, max_len))
    for s_id, sentence in enumerate(forward):
        for w_id, word in enumerate(sentence):
            forward_ground[s_id][w_id] = word2id.get(word, 1)
    for s_id, sentence in enumerate(backward):
        for w_id, word in enumerate(sentence):
            backward_ground[s_id][w_id] = word2id.get(word, 1)
    return forward_ground, backward_ground
    



def token_elmo(sentences, word2id):
    """
    token: raw data, list of sentence, sentence has different length
    for example: [["i", "love", "you"], ["the", "sky", "is", "blue"]] -> [["<s>", i", "love", "you", "</s>"], ["<s>", "the", "sky", "is", "blue", "</s>"]]
    padding for sentence is 0, so utf-8 need + 1
    res is batch, max_seq_len, 50

    forward: [["<s>", i", "love", "you"], ["<s>", "the", "sky", "is", "blue"]]
    backward:[[</s>, you, love, I], ["</s>", "blue", "is", "sky", "the"]]



    return: res and word mask
    """
    forward, backward = [], []
    for sentence in sentences:
        forward.append(["<s>"]+sentence)
        backward.append(sentence+["</s>"])
    backward = [list(reversed(sentence)) for sentence in backward]
    forward_res, forward_mask = token2id(forward)
    backward_res, backward_mask = token2id(backward)
    forward_ground, backward_ground = build_ground(sentences, word2id)
    return forward_res, forward_mask, forward_ground, backward_res, backward_mask, backward_ground

def build_word2id(filename):
    with open(filename, "r") as f:
        wordlist = json.load(f)
    word2id = {}
    word2id["<pad>"] = 0
    word2id["<unk>"] = 1
    word2id["<s>"] = 2
    word2id["</s>"] = 3
    idx = 4
    for word in wordlist:
        word2id[word] = idx
        idx += 1
    return word2id

def batch_generator(data, batch_size):
    for start in range(0, len(data), batch_size):
        yield(data[start: start+batch_size])

    

if __name__ == "__main__":
    base_dir = "/home/FuDawei/NLP/Pretrained_model/dataset/"
    word2id = build_word2id(base_dir+"wordlist.json")
    with open(base_dir+"word2id.json", "w") as f:
        json.dump(word2id, f)
    











