import torch
import ujson as json
import pickle as pkl
import torch.nn as nn

from util.data_util import get_data, get_batch_data
from model import Unet
from util.evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
def prepare_train(save_dir):
    train_examples = get_data(save_dir+"train.json")
    dev_examples = get_data(save_dir+"dev.json")

    opt = {}
    opt["word_emb_path"] = save_dir + 'word_emb_matrix.json'
    opt["ner2id_path"] = save_dir + 'ner2id.json'
    # word2id = get_data(save_dir + 'word2id.json')
    opt["tune_idx_path"] = save_dir + "tune_idx.json"
    opt["pos2id_path"] = save_dir + 'pos2id.json'
    opt["word_dim"], opt["ner_dim"], opt["pos_dim"], opt["elmo_dim"] = 100, 12, 8, 1024
    opt["drop_rate"] = 0.3
    opt["hidden_size"] = 125
    opt["biattention_size"] = 250
    opt["lr"] = 0.002
    opt["use_elmo"] = False
    opt["fix_word_embedding"] = False
    opt["save_dir"] = save_dir

    return train_examples, dev_examples, opt

def train():
    save_dir = "/home/FuDawei/NLP/SQUAD/unet/data/"
    train_examples, dev_examples, opt = prepare_train(save_dir)
    epoch = 30
    batch_size = 32
    model = Unet(opt=opt).to(device)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adamax(parameters, lr = opt["lr"])
    best_score, exact_scores, f1_scores = 0, [], []

    count = 0
    total_loss = 0
    for ep in range(epoch):
        model.train()
        for batch_data in get_batch_data(train_examples, batch_size):
            data = model.get_data(batch_data)
            loss = model(data)
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 10)
            optimizer.step()
            model.reset_parameters()
            count += 1
            # print(loss.item())
            # Evaluate(dev_examples, model)

            total_loss += loss.item()
            if count % 100 == 0:
                print(total_loss/100)
                total_loss = 0
                # model.eval()
                # Evaluate(dev_examples, model, opt)
            if not opt["fix_word_embedding"]:
                model.reset_parameters()
        print(ep)  
        model.eval()
        exact, f1 = Evaluate(dev_examples, model, opt)
        exact_scores.append(exact)
        f1_scores.append(f1)
        if f1 > best_score:
            best_score = f1
            torch.save(model.state_dict(), save_dir+"best_model")
    with open(save_dir + '_f1_scores.pkl', 'wb') as f:
        pkl.dump(f1_scores, f)
    with open(save_dir + '_exact_scores.pkl', 'wb') as f:
        pkl.dump(exact_scores, f)


def Evaluate(data, model, opt):
    """
    data : dev_examples
    """
    save_dir = opt["save_dir"]
    dev_preds = {}
    dev_na_probs = {}
    batch_size = 32
    flag = True
    for batch_data in get_batch_data(data, batch_size):

        context_tokens = batch_data["context_tokens"]
        data = model.get_data(batch_data)
        ids = data["ids"]
        starts, ends, answer_prob = model.prediction(data, flag)
        flag = False
        answer_context = []
        # print(starts)
        # print(ends)
        # print(context_tokens)
        for idx in range(len(starts)):
            start = starts[idx]
            end = ends[idx]
            context_token = context_tokens[idx]
            if (start == 0 and end == 0) or start > end:
                answer_context.append("")
            else:
                answer_context.append(" ".join(context_token[start:end+1]))
        for idx in range(len(ids)):
            id = ids[idx]
            answer = answer_context[idx]
            no_prob = 1.0 - answer_prob[idx].item()
            dev_preds[id] = answer
            dev_na_probs[id] = no_prob
    with open(save_dir+"dev_preds","w", encoding="utf-8") as f:
        json.dump(dev_preds, f)
    dev_file_path = "/home/FuDawei/NLP/SQUAD/data/dev-v2.0.json"

    ignore_ids = []
    with open(save_dir+"devdrop_in_process.json", 'r') as f :
        dev_drop_process = json.load(f)
        ignore_ids.extend(dev_drop_process["id"])

    with open(save_dir+"dev_drop.json", 'r') as f :
        dev_drop = json.load(f)
        ignore_ids.extend(dev_drop["id"])

    out_eval = evaluate(dev_file_path, dev_preds, na_prob_thresh=0.3,ignore_ids = ignore_ids)
    exact, f1 = out_eval["exact"], out_eval["f1"]
    return exact, f1


    
        

            

        

if __name__ == "__main__":
    train()