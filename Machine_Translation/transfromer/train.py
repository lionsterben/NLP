import sys
sys.path.append("/home/FuDawei/NLP/Machine_Translation")
from model import Transformer
from data_util.util import generate_batch, get_origin_data
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import ujson as json
from util import optimizer_lr_rate

device = torch.device("cuda:0")
PAD_ID = 0
UNK_ID = 1
EOS_ID = 2
SOS_ID = 3

def main():
    # source_vocab_size, target_vocab_size, d_model, key_hidden_size, value_hidden_size, heads, ff_size, drop_rate = 0
    opt = {}
    opt["source_vocab_size"] = 50000 + 4
    opt["target_vocab_size"] = 50000 + 4
    opt["d_model"] = 512
    opt["key_hidden_size"] = 64
    opt["value_hidden_size"] = 64
    opt["heads"] = 8
    opt["ff_size"] = 2048
    opt["drop_rate"] = 0.1
    opt["epoch"] = 30
    opt["lr_rate"] = 0.001

    train(opt)


def train(opt):
    source_vocab_size = opt["source_vocab_size"]
    target_vocab_size = opt["target_vocab_size"]
    d_model = opt["d_model"]
    key_hidden_size = opt["key_hidden_size"]
    value_hidden_size = opt["value_hidden_size"]
    heads = opt["heads"]
    ff_size = opt["ff_size"]
    drop_rate = opt["drop_rate"]
    epoch = opt["epoch"]
    lr_rate = opt["lr_rate"]
    model = Transformer(source_vocab_size, target_vocab_size, d_model, key_hidden_size, value_hidden_size, heads,
                        ff_size, drop_rate).to(device)
    base_dir = "/home/FuDawei/NLP/Machine_Translation/dataset/en-fr-no/"
    source_lines, source_word2id, source_id2word = get_origin_data(base_dir+"frtrain_lines", base_dir+"fr_word2id.json", base_dir+"fr_id2word.json")
    target_lines, target_word2id, target_id2word = get_origin_data(base_dir+"entrain_lines", base_dir+"en_word2id.json", base_dir+"en_id2word.json")
    parameters = filter(lambda i: i.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=lr_rate)
    print_every = 1000
    cnt = 0
    total = 0
    save_dir = "/home/FuDawei/NLP/Machine_Translation/dataset/save_res/"
    batch_size = 64
    max_len = 50
    for ep in range(opt["epoch"]):
        for start_index in range(0, len(source_lines), batch_size):
            source_input, _, source_input_token, _ = generate_batch(source_lines, source_word2id, source_id2word,
                                                                    batch_size, start_index, False, True, max_len)
            target_input, target_output, target_input_token, target_output_token = generate_batch(target_lines,
                                                                                                  target_word2id,
                                                                                                  target_id2word,
                                                                                                  batch_size,
                                                                                                  start_index, True,
                                                                                                  True, max_len)
            source_input = torch.tensor(source_input).to(device)
            target_input, target_output = torch.tensor(target_input).to(device), torch.tensor(target_output).to(device)
            decoder_logits = model(source_input, target_input)
            loss = model.compute_loss(decoder_logits, target_output)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 5)
            optimizer.step()
            cnt += 1
            total += loss.item()
            if cnt % print_every == 0:
                print(total / 1000)
                total = 0
            if cnt % 10000 == 0:
                model.eval()
                preds, targets = get_prediction(base_dir + "frdev_lines", base_dir + "endev_lines", model)
                with open(save_dir + str(cnt) + "origin_transformer_preds.json", "w") as f:
                    json.dump(preds, f)
                with open(save_dir + str(cnt) + "origin_transformer_targets.json", "w") as f:
                    json.dump(targets, f)
                # print(eval(preds, targets))

                model.train()
        # if ep > 0 and ep % 2 == 0:
        #     lr_rate = optimizer_lr_rate(d_model, cnt)
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = lr_rate
        print(ep)
        lr_rate = lr_rate * 0.5
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_rate



def get_prediction(source_path, target_path, model):
    # use beam search
    preds = []
    targets = []
    max_len = 50
    base_dir = "/home/FuDawei/NLP/Machine_Translation/dataset/en-fr-no/"
    source_lines, source_word2id, source_id2word = get_origin_data(source_path, base_dir+"fr_word2id.json", base_dir+"fr_id2word.json")
    target_lines, target_word2id, target_id2word = get_origin_data(target_path, base_dir+"en_word2id.json", base_dir+"en_id2word.json")
    for start_index in range(0, len(source_lines), 32):
        source_input, _, source_input_token, _ = generate_batch(source_lines, source_word2id, source_id2word, 32, start_index, False, True, max_len)
        target_input, target_output, target_input_token, target_output_token = generate_batch(target_lines, target_word2id, target_id2word, 32, start_index, True, True, max_len)
        source_input = torch.tensor(source_input).to(device)
        batch_size = source_input.size(0)
        # target_input, target_output = torch.tensor(target_input).to(device), torch.tensor(target_output).to(device)
        batch_preds = [[] for _ in range(batch_size)]
        mask = [0 for _ in range(batch_size)]
        encoder_output = model.encode_forward(source_input)
        decoder_input = torch.tensor(SOS_ID).repeat(batch_size).view(batch_size, 1).to(device)
        for _ in range(max_len):
            output_logits = model.decode_forward(decoder_input, encoder_output) #batch_size, seq_len, vocab_size
            output_logits = output_logits[:, -1]
            output_id  = torch.argmax(output_logits, 1).tolist() # batch_size
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
            decoder_input = torch.cat([decoder_input, torch.tensor(output_id, device=device).view(batch_size, 1).to(device)], 1)
        preds.extend(batch_preds)
        targets.extend(target_output_token)
    print(preds[:2])
    print(targets[:2])
    return preds, targets


if __name__ == "__main__":
    main()




