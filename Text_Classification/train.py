import ujson as json
import torch
from model import DPCNN
from util import batch_generator
from sklearn.metrics import confusion_matrix
from torch import optim

device = torch.device("cuda:1")

def main():
    opt = {}
    opt["hidden_size"] = 300
    opt["feature_map"] = 250
    opt["seq_len"] = 100
    opt["num_class"] = 5
    opt["vocab_size"] = 30000+2
    opt["drop_rate"] = 0.5
    opt["epoch"] = 10
    opt["batch_size"] = 128
    opt["lr"] = 0.001
    train(opt)

def train(opt):
    model = DPCNN(opt["hidden_size"], opt["feature_map"], opt["seq_len"], opt["num_class"], opt["vocab_size"], opt["drop_rate"]).to(device)
    with open("/home/FuDawei/NLP/Text_Classification/dataset/train_text.json", "r") as f:
        train_text = json.load(f)
    with open("/home/FuDawei/NLP/Text_Classification/dataset/train_star.json", "r") as f:
        train_star = json.load(f)
    with open("/home/FuDawei/NLP/Text_Classification/dataset/dev_text.json", "r") as f:
        dev_text = json.load(f)
    with open("/home/FuDawei/NLP/Text_Classification/dataset/dev_star.json", "r") as f:
        dev_star = json.load(f)
    
    
    optimizer = optim.Adam(model.parameters(), opt["lr"])
    cnt = 0
    total_loss = 0

    for ep in range(opt["epoch"]):
        for text, star in batch_generator(train_text, train_star, opt["batch_size"]):
            text, star = torch.tensor(text).to(device), torch.tensor(star).to(device)
            logits = model(text)
            loss = model.compute_loss(logits, star)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1
            total_loss += loss.item()
            if cnt%100 == 0:
                print(total_loss/100)
                total_loss = 0
            if cnt%1000 == 0:
                model.eval()
                preds = []
                for text, _ in batch_generator(dev_text, dev_star, opt["batch_size"]):
                    text = torch.tensor(text).to(device)
                    logits = model(text)
                    pred = model.compute_res(logits).tolist()
                    preds.extend(pred)
                a = confusion_matrix(dev_star, preds)
                print(a)
                right, all = 0, 0
                for idx, item in enumerate(a):
                    right += item[idx]
                    all += sum(item)
                final_rate = right/all
                print(final_rate)
                model.train()

if __name__ == "__main__":
    main()



            

            


