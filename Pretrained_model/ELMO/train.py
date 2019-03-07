import torch
import ujson as json
from model import Elmo
from util import batch_generator, token_elmo
from torch import optim

device = torch.device("cuda:1")
# device = torch.device("cpu")

def main():
    opt = {}
    opt["char_size"] = 262
    opt["char_emb_size"] = 256
    opt["embedding_size"] = 256
    opt["hidden_size"] = 256
    opt["vocab_size"] = 10000+4
    opt["drop_rate"] = 0.3
    opt["epoch"] = 10
    opt["batch_size"] = 32
    opt["lr"] = 0.001
    train(opt)

def train(opt):
    base_dir = "/home/FuDawei/NLP/Pretrained_model/dataset/"
    model = Elmo(opt["char_size"], opt["char_emb_size"], opt["embedding_size"], opt["hidden_size"], opt["vocab_size"], opt["drop_rate"]).to(device)
    with open(base_dir+"word2id.json", "r") as f:
        word2id = json.load(f)
    with open(base_dir+"elmo_lower_data.json", "r") as f:
        data = json.load(f)
    optimizer = optim.Adam(model.parameters(), lr=opt["lr"])
    cnt = 0
    lo = 0
    
    for ep in range(opt["epoch"]):
        for batch_data in batch_generator(data, opt["batch_size"]):
            forward_res, forward_mask, forward_ground, backward_res, backward_mask, backward_ground = token_elmo(batch_data, word2id)
            forward_input, forward_mask, forward_ground, backward_input, backward_mask, backward_ground = \
                torch.tensor(forward_res).long().to(device), torch.tensor(forward_mask).to(device), torch.tensor(forward_ground).long().to(device), \
                torch.tensor(backward_res).long().to(device), torch.tensor(backward_mask).to(device), torch.tensor(backward_ground).long().to(device)
            forward_output, backward_output = model(forward_input, forward_mask, backward_input, backward_mask)
            loss = model.compute_loss(forward_output, backward_output, forward_ground, backward_ground)
            # print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1
            lo += loss.item()
            if cnt % 100 == 0:
                print(lo/100)
                lo = 0

if __name__ == "__main__":
    main()
            
    
