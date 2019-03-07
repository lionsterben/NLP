from model import Basic_NMT
import sys
sys.path.append("/home/FuDawei/NLP/Machine_Translation")
from data_util.util import generate_batch, get_origin_data
import torch
from torch import optim
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import ujson as json
import torch.nn as nn

device = torch.device("cuda:0")
PAD_ID = 0
UNK_ID = 1
EOS_ID = 2
SOS_ID = 3

# def eval(preds, truths):
#     """pred:list of sentences, sentence is list of words"""
#     res_score = 0
#     for idx in range(len(preds)):
#         pred, truth = preds[idx], truths[idx]
#         score = sentence_bleu([truth], pred)
#         res_score += score
#     return res_score/len(preds)

def main():
    opt = {}
    opt['learning_rate'] = 0.001
    opt["batch_size"] = 80
    opt["source_vocab_size"] = 50000+4
    opt["target_vocab_size"] = 50000+4
    opt["max_len"] = 50
    opt["source_hidden_size"] = 1000
    opt["target_hidden_size"] = 1000
    opt["source_emb_size"] = 1024
    opt["target_emb_size"] = 1024
    opt["epoch"] = 30
    train(opt)

def compute_mask(ids):
    """
    ids: (batch_size, length)
    0 is padding
    """
    return ids.ne(0).int()

def train(opt):
    source_vocab_size = opt["source_vocab_size"]
    target_vocab_size = opt["target_vocab_size"]
    source_hidden_size = opt["source_hidden_size"]
    target_hidden_size = opt["target_hidden_size"]
    source_emb_size = opt["source_emb_size"]
    target_emb_size = opt["target_emb_size"]
    batch_size = opt["batch_size"]
    max_len = opt["max_len"]
    model = Basic_NMT(source_vocab_size, target_vocab_size, source_emb_size, target_emb_size, source_hidden_size, source_hidden_size).to(device)
    base_dir = "/home/FuDawei/NLP/Machine_Translation/dataset/en-fr-no/"
    source_lines, source_word2id, source_id2word = get_origin_data(base_dir+"frtrain_lines", base_dir+"fr_word2id.json", base_dir+"fr_id2word.json")
    target_lines, target_word2id, target_id2word = get_origin_data(base_dir+"entrain_lines", base_dir+"en_word2id.json", base_dir+"en_id2word.json")
    # source_lines, source_word2id, source_id2word = get_origin_data("/home/FuDawei/NLP/Machine_Translation/dataset/debug/endebug", base_dir+"en_word2id.json", base_dir+"en_id2word.json")
    # target_lines, target_word2id, target_id2word = get_origin_data("/home/FuDawei/NLP/Machine_Translation/dataset/debug/frdebug", base_dir+"fr_word2id.json", base_dir+"fr_id2word.json")
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=opt["learning_rate"])
    print_every = 1000
    cnt = 0
    total = 0
    save_dir = "/home/FuDawei/NLP/Machine_Translation/dataset/save_res/"
    for ep in range(opt["epoch"]):
        for start_index in range(0, len(source_lines), batch_size):
            source_input, _, source_input_token, _ = generate_batch(source_lines, source_word2id, source_id2word, batch_size, start_index, False, True, max_len)
            target_input, target_output, target_input_token, target_output_token = generate_batch(target_lines, target_word2id, target_id2word, batch_size, start_index, True, True, max_len)
            # print(source_input_token[3])
            # print(target_input_token[3])
            # print(target_output_token[3])

            # # assert 1==0
            # valid_idx = []
            # for idx in range(len(source_input)):
            #     if sum(source_input[idx]) != 0:
            #         valid_idx.append(idx)
            # source_input = np.array(source_input)[valid_idx]
            # target_input = np.array(target_input)[valid_idx]
            # target_output = np.array(target_output)[valid_idx]

            source_input = torch.tensor(source_input).to(device)
            source_input_mask = compute_mask(source_input)
            target_input, target_output = torch.tensor(target_input).to(device), torch.tensor(target_output).to(device)
            decoder_logits = model(source_input, source_input_mask, target_input)
            loss = model.compute_loss(decoder_logits, target_output)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 10)
            optimizer.step()
            cnt += 1
            total += loss.item()
            if cnt % print_every == 0:
                print(total/1000)
                total = 0
            if cnt % 10000 == 0:
                # topv, topi = decoder_logits.data.topk(1)
                # topi = topi.squeeze(1)
                # for idx in range(5):
                #     pred = ""
                #     real = ""
                #     for i in range(target_output.size(1)):
                #         if target_id2word[str(target_output[idx][i].item())] == "<pad>":
                #             break
                #         real = real + target_id2word[str(target_output[idx][i].item())]+ " "
                #         # real.append(target_id2word[str(target_output[idx][i].item())])
                #         pred = pred + target_id2word[str(topi[idx][i].item())] + " "
                #         # pred.append(target_id2word[str(topi[idx][i].item())])
                
                #     print(pred)
                #     print("\n")
                #     print(real)
                #     print("\n")
                        
                model.eval()
                preds, targets = get_prediction(base_dir+"frdev_lines", base_dir+"endev_lines", model)
                with open(save_dir+str(cnt)+"preds.json", "w") as f:
                    json.dump(preds, f)
                with open(save_dir+str(cnt)+"targets.json", "w") as f:
                    json.dump(targets, f)
        # print(eval(preds, targets))
        
                model.train()

def get_prediction(source_path, target_path, model):
    # use beam search
    preds = []
    targets = []
    i = 0
    batch_size = 32
    max_len = 50
    base_dir = "/home/FuDawei/NLP/Machine_Translation/dataset/en-fr-no/"
    source_lines, source_word2id, source_id2word = get_origin_data(source_path, base_dir+"fr_word2id.json", base_dir+"fr_id2word.json")
    target_lines, target_word2id, target_id2word = get_origin_data(target_path, base_dir+"en_word2id.json", base_dir+"en_id2word.json")
    for start_index in range(0, len(source_lines), 32):
        source_input, _, source_input_token, _ = generate_batch(source_lines, source_word2id, source_id2word, 32, start_index, True, True, max_len)
        target_input, target_output, target_input_token, target_output_token = generate_batch(target_lines, target_word2id, target_id2word, 32, start_index, True, True, max_len)
        source_input = torch.tensor(source_input).to(device)
        batch_size = source_input.size(0)
        source_input_mask = compute_mask(source_input)
        target_input, target_output = torch.tensor(target_input).to(device), torch.tensor(target_output).to(device)
        batch_preds = [[] for _ in range(batch_size)]
        mask = [0 for _ in range(batch_size)]
        encode_state = model.encode(source_input, source_input_mask)
        decode_input = torch.tensor(SOS_ID).repeat(batch_size).view(batch_size, 1).to(device)
        # print(hidden_cell_state)
        for _ in range(max_len):
            output_logits, decoder_state = model.decode(decode_input, encode_state) #batch_size, 1, vocab_size
            output_logits = output_logits.squeeze(1)
            output_id  = torch.argmax(output_logits, 1).tolist() # batch_size
            # topv, topi = output_logits.data.topk(1)
            # topi = topi.squeeze(1)
            for idx in range(batch_size):
                if mask[idx]:
                    continue
                id = output_id[idx]
                if id == EOS_ID:
                    mask[idx] = 1
                else:
                    batch_preds[idx].append(target_id2word[str(id)])
            if sum(mask) == len(mask):
                break
            decode_input = torch.tensor(output_id, device = device).view(batch_size, 1).to(device)
            encode_state = decoder_state
        preds.extend(batch_preds)
        targets.extend(target_output_token)
    print(preds[:2])
    print(targets[:2])
    return preds, targets

if __name__ == "__main__":
    main()


            

