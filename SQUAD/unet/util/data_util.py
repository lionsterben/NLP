import torch
import ujson as json
from allennlp.modules.elmo import Elmo


def compute_mask_old(x):
    """
    ids: (batch_size, length)
    1 is padding
    """
    return torch.eq(x, 0)

def compute_mask(ids):
    """
    ids: (batch_size, length)
    0 is padding
    """
    return ids.ne(0).int()

def get_batch_data(data, batch_size):
    length = len(data["context_ids"])
    for batch_start in range(0, length, batch_size):
        res = {}
        size = batch_size
        if batch_start+batch_size > length:
            size = length-batch_start
        for key in data.keys():
            # print(key)
            # print(data[key])
            if key != "total":
                res[key] = data[key][batch_start: batch_start+size]  
        yield res

def get_data(filename) :
    with open(filename, 'r', encoding='utf-8') as f :
        data = json.load(f)
    return data


    

