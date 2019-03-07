## use allennlp env , elmo is optional
## context token是没有padding的，所以在一个batch里需要再次求一下maxlen，进行阶段，ques不用

## word Embedding need fix, pos and ner need self.training
## binary feature 3
## word_dim + elmo + ner_dim + pos_dim + 3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ujson as json

from data_precess import data_from_json
from util.layer import Bilstm, BasicAttention, BiAttention, AnswerPoint, masked_softmax
from util.data_util import compute_mask
from allennlp.modules.elmo import Elmo
from allennlp.modules.elmo import batch_to_ids


option = "/home/FuDawei/NLP/Embedding/English/elmo/elmo_2x1024_128_2048cnn_1xhighway_options.json"
weight = "/home/FuDawei/NLP/Embedding/English/elmo/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

class Unet(nn.Module):
    def __init__(self, opt):
        """opt conf file"""
        super(Unet, self).__init__()
        self.opt = opt
        self.use_elmo = opt["use_elmo"]
        self.build_model()

    def build_model(self):
        ## fix 3:end word embedding,
        opt = self.opt
        word_emb = np.array(data_from_json(opt["word_emb_path"]))
        ner2id = data_from_json(opt["ner2id_path"])
        pos2id = data_from_json(opt["pos2id_path"])
        tune_idx = data_from_json(opt["tune_idx_path"])
        word_dim, ner_dim, pos_dim = opt["word_dim"], opt["ner_dim"], opt["pos_dim"]
        word_vocab_size, ner_size, pos_size = word_emb.shape[0], len(ner2id), len(pos2id)

        self.word_embedding = nn.Embedding(word_vocab_size, word_dim, padding_idx=0)
        self.word_embedding.weight.data.copy_(torch.from_numpy(word_emb))
        # self.fixed_idx = [i for i in range(3, word_vocab_size)]
        # fixed_embedding = torch.from_numpy(word_emb)[self.fixed_idx]
        # self.register_buffer('fixed_embedding', fixed_embedding)
        # self.fixed_embedding = fixed_embedding
        # self.word_embedding.weight.requires_grad = False
        if not opt["fix_word_embedding"]:
            self.fixed_idx = list(set([i for i in range(word_vocab_size)]) - set(tune_idx))
            fixed_embedding = torch.from_numpy(word_emb)[self.fixed_idx]
            self.register_buffer('fixed_embedding', fixed_embedding)
            self.fixed_embedding = fixed_embedding
        # self.word_embedding.weight.requires_grad = False
        else:
            for p in self.word_embedding.parameters():
                p.requires_grad = False

        self.ner_embedding = nn.Embedding(ner_size, ner_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(pos_size, pos_dim, padding_idx=0)

        drop_rate = opt["drop_rate"]
        hidden_size = opt["hidden_size"]

        # binary feature: match, lemma match, tf 
        binary_dim = 3
        input_dim = word_dim + ner_dim + pos_dim + 3
        if self.use_elmo:
            input_dim += opt["elmo_dim"]
            self.elmo = Elmo(option, weight, 1, dropout=0)
        
        ## word attention
        self.ques_attention = BasicAttention(drop_rate, word_dim, word_dim) # key is question, just use word dimension
        self.context_attention = BasicAttention(drop_rate, word_dim, word_dim) # key is context, just use word dimension

        input_dim += word_dim # for word attention

        ## 3 layer bilstm
        output_size = 2*hidden_size
        self.low_bilstm = Bilstm(input_dim, hidden_size, drop_rate) 
        self.middle_bilstm = Bilstm(output_size, hidden_size, drop_rate)
        ## high bilstm use low output and middle output
        self.high_bilstm = Bilstm(2*output_size, hidden_size, drop_rate)
        # final_lstm_size = 3*output_size

        ## biattention layer for three lstm
        biattention_size = opt["biattention_size"]
        self.low_biattention = BiAttention(drop_rate, output_size, biattention_size)
        self.middle_biattention = BiAttention(drop_rate, output_size, biattention_size)
        self.high_biattention = BiAttention(drop_rate, output_size, biattention_size)

        ##for final fusion ,3 layer bilstm + 3 layer biattention
        self.final_fusion_lstm = Bilstm(6*output_size, hidden_size, drop_rate)

        ## self attention layer, input is origin input plus final fusion
        self.self_attention = BasicAttention(drop_rate, input_dim+2*hidden_size, input_dim+2*hidden_size)

        ## last bilstm layer, input is HA plus self attention output
        self.last_bilstm = Bilstm(input_dim+2*hidden_size+2*hidden_size, hidden_size, drop_rate)

        final_fusion_size = 2*hidden_size

        self.ques_linear = nn.Linear(final_fusion_size, 1, bias=False)
        self.answer_verifier_linear = nn.Linear(4*final_fusion_size, 2)

        ## answer layer
        self.true_answer_pointer = AnswerPoint(final_fusion_size, final_fusion_size)
        self.fake_answer_pointer = AnswerPoint(final_fusion_size, final_fusion_size)

        self.verifier_loss = nn.BCEWithLogitsLoss()
    
    def reset_parameters(self):
        self.word_embedding.weight.data[self.fixed_idx] = self.fixed_embedding.float()

    
    

    def get_data(self, batch_data):
    #         data = {
    #     "context_ids" : context_ids,
    #     "context_tokens" : context_tokens, context elmo ids ,will pad to max_len
    #     "context_matchs" : context_matchs,
    #     "context_lemma_matchs" : context_lemma_matchs,
    #     "context_tfs" : context_tfs,
    #     "context_pos_ids" : context_pos_ids,
    #     "context_ner_ids" : context_ner_ids,
    #     "ques_ids" : ques_ids,
    #     "ques_matchs" : ques_matchs,
    #     "ques_lemma_matchs" : ques_lemma_matchs,
    #     "ques_tokens" : ques_tokens, ques elmo ids
    #     "ques_pos_ids" : ques_pos_ids,
    #     "ques_ner_ids" : ques_ner_ids,
    #     "ques_tfs": ques_tfs,
    #     "true_starts" : true_starts,
    #     "true_ends" : true_ends,
    #     "fake_starts" : fake_starts,
    #     "fake_ends" : fake_ends,
    #     "ids" : ids,
    #     "total" : count_without_drop,
    #     "has_ans": has_ans
    # }
        context_ids = torch.tensor(batch_data["context_ids"], device=device)
        context_mask = compute_mask(context_ids)
        context_max_len = int(torch.max(torch.sum(context_mask, 1)).item())

        context_ids = torch.tensor(batch_data["context_ids"], device = device)[:, :context_max_len]
        context_matchs = torch.tensor(batch_data["context_matchs"], device = device)[:, :context_max_len]
        context_lemma_matchs = torch.tensor(batch_data["context_lemma_matchs"], device = device)[:, :context_max_len]
        context_tfs = torch.tensor(batch_data["context_tfs"], device = device)[:, :context_max_len]
        context_pos_ids = torch.tensor(batch_data["context_pos_ids"], device = device)[:, :context_max_len]
        context_ner_ids = torch.tensor(batch_data["context_ner_ids"], device = device)[:, :context_max_len]

        ques_ids = torch.tensor(batch_data["ques_ids"], device=device)
        ques_mask = compute_mask(ques_ids)
        ques_max_len = int(torch.max(torch.sum(ques_mask, 1)).item())
        ques_limit = len(batch_data["ques_tfs"][0])

        ques_ids = torch.tensor(batch_data["ques_ids"], device = device)[:, ques_limit-ques_max_len:]
        ques_matchs = torch.tensor(batch_data["ques_matchs"], device = device)[:, ques_limit-ques_max_len:]
        ques_lemma_matchs = torch.tensor(batch_data["ques_lemma_matchs"], device = device)[:, ques_limit-ques_max_len:]
        ques_tfs = torch.tensor(batch_data["ques_tfs"], device = device)[:, ques_limit-ques_max_len:]
        ques_pos_ids = torch.tensor(batch_data["ques_pos_ids"], device = device)[:, ques_limit-ques_max_len:]
        ques_ner_ids = torch.tensor(batch_data["ques_ner_ids"], device = device)[:, ques_limit-ques_max_len:]

        if self.use_elmo:
            ques_tokens = batch_data["ques_tokens"][:, ques_limit-ques_max_len:, :]
            context_tokens = batch_data["context_tokens"]

        true_starts = torch.tensor(batch_data["true_starts"], device=device)
        # print(true_starts)
        true_ends = torch.tensor(batch_data["true_ends"], device=device)
        # print(true_ends)
        fake_starts = torch.tensor(batch_data["fake_starts"], device=device)
        fake_ends = torch.tensor(batch_data["fake_ends"], device=device)
        has_ans = torch.tensor(batch_data["has_ans"], device=device)

        # context_mask = compute_mask(context_ids)
        # ques_mask = compute_mask(ques_ids)

        data = {
            "context_ids" : context_ids,
            "context_matchs" : context_matchs.float(),
            "context_lemma_matchs" : context_lemma_matchs.float(),
            "context_tfs" : context_tfs,
            "context_pos_ids" : context_pos_ids,
            "context_ner_ids" : context_ner_ids,
            "ques_ids" : ques_ids,
            "ques_matchs" : ques_matchs.float(),
            "ques_lemma_matchs" : ques_lemma_matchs.float(),
            "ques_pos_ids" : ques_pos_ids,
            "ques_ner_ids" : ques_ner_ids,
            "ques_tfs": ques_tfs,
            "true_starts" : true_starts,
            "true_ends" : true_ends,
            "fake_starts" : fake_starts,
            "fake_ends" : fake_ends,
            "ids" : batch_data["ids"],
            "has_ans": has_ans,
            "ques_max_len": ques_max_len,
            "context_max_len": context_max_len,
            "cnt": batch_data["cnts"]
        }

        if self.use_elmo:
            data["context_tokens"] = context_tokens
            data["ques_tokens"] = ques_tokens

        return data

    def encode_forward(self, data):
        batch_size = data["context_ids"].size()[0]
        context_ids = data["context_ids"]
        context_mask = compute_mask(context_ids)
        context_word_embedding = self.word_embedding(context_ids)
        ques_ids = data["ques_ids"]
        ques_mask = compute_mask(ques_ids)
        ques_word_embedding = self.word_embedding(ques_ids)

        ques_word_attention = self.ques_attention(context_word_embedding, context_mask, ques_word_embedding)
        context_word_attention = self.context_attention(ques_word_embedding, ques_mask, context_word_embedding)
        para_word_attention = torch.cat([ques_word_attention, context_word_attention], 1)

        para_ids = torch.cat((ques_ids, context_ids), 1)
        para_mask = torch.cat((ques_mask, context_mask), 1)
        para_word_embedding = self.word_embedding(para_ids)
        # print(para_mask)
        # print(data["ques_max_len"])
        # print(ques_mask.size())
        # assert 1 == 0

        context_matchs = data["context_matchs"]
        ques_matchs = data["ques_matchs"]
        para_matchs = torch.cat((ques_matchs, context_matchs), 1).unsqueeze(2) # (batch_size, para_len, 1)

        context_lemma_matchs = data["context_lemma_matchs"]
        ques_lemma_matchs = data["ques_lemma_matchs"]
        para_lemma_matchs = torch.cat((ques_lemma_matchs, context_lemma_matchs), 1).unsqueeze(2) # (batch_size, para_len, 1)

        context_tfs = data["context_tfs"]
        ques_tfs = data["ques_tfs"]
        para_tfs = torch.cat((ques_tfs, context_tfs), 1).unsqueeze(2) # (batch_size, para_len, 1)

        ques_ner_ids = data["ques_ner_ids"]
        context_ner_ids = data["context_ner_ids"]
        para_ner_ids = torch.cat((ques_ner_ids, context_ner_ids), 1)
        para_ner_embedding = self.ner_embedding(para_ner_ids)

        ques_pos_ids = data["ques_pos_ids"]
        context_pos_ids = data["context_pos_ids"]
        para_pos_ids = torch.cat((ques_pos_ids, context_pos_ids), 1)
        para_pos_embedding = self.pos_embedding(para_pos_ids)

        if self.use_elmo:
            ques_tokens = data["ques_tokens"]
            context_tokens = data["context_tokens"]
            ques_elmo_id = batch_to_ids(ques_tokens)
            ques_elmo_embedding = self.elmo(ques_elmo_id)["elmo_representations"][0]
            context_elmo_id = batch_to_ids(context_tokens)
            context_elmo_embedding = self.elmo(context_elmo_id)["elmo_representations"][0]
            para_elmo_embedding = torch.cat([ques_elmo_embedding, context_elmo_embedding], 1)
        
        input = torch.cat([para_word_embedding, para_word_attention, para_matchs, para_lemma_matchs, para_tfs, para_ner_embedding, para_pos_embedding], 2)
        if self.use_elmo:
            input = torch.cat([input, para_elmo_embedding], 2)
        
        ## now is 3 bilstm
        ignore_length = data["ques_max_len"]
        # print(para_mask)
        low_lstm_output, _ = self.low_bilstm(input, para_mask, ignore_length)
        middle_lstm_output, _ = self.middle_bilstm(low_lstm_output, para_mask, ignore_length)
        high_lstm_output, _ = self.high_bilstm(torch.cat([low_lstm_output, middle_lstm_output], 2), para_mask, ignore_length)

        lstm_ouput = torch.cat([low_lstm_output, middle_lstm_output, high_lstm_output], 2)

        ## now is biattention
        ques_len = data["ques_max_len"] 
        context_len = data["context_max_len"] # include noanswer node
        low_lstm_ques, low_lstm_context = low_lstm_output[:, :ques_len+1,:], low_lstm_output[:, ques_len:, :]
        middle_lstm_ques, middle_lstm_context = middle_lstm_output[:, :ques_len+1,:], middle_lstm_output[:, ques_len:, :]
        high_lstm_ques, high_lstm_context = high_lstm_output[:, :ques_len+1,:], high_lstm_output[:, ques_len:, :]
        mask_for_node = torch.ones(batch_size, 1).int().to(device)
        ques_mask_for_attention = torch.cat([ques_mask, mask_for_node], 1)
        context_mask_for_attention = context_mask

        low_attention_context, low_attention_ques = self.low_biattention(low_lstm_ques, low_lstm_context, ques_mask_for_attention, context_mask_for_attention)
        middle_attention_context, middle_attention_ques = self.middle_biattention(middle_lstm_ques, middle_lstm_context, ques_mask_for_attention, context_mask_for_attention)
        high_attention_context, high_attention_ques = self.high_biattention(high_lstm_ques, high_lstm_context, ques_mask_for_attention, context_mask_for_attention)
        
        ## ques_attention, context_attention have unify node, so we need add this node
        low_attention_context[:, 0] += low_attention_ques[:, ques_len]
        low_attention_para = torch.cat([low_attention_ques[:, :ques_len], low_attention_context], 1)

        middle_attention_context[:, 0] += middle_attention_ques[:, ques_len]
        middle_attention_para = torch.cat([middle_attention_ques[:, :ques_len], middle_attention_context], 1)

        high_attention_context[:, 0] += high_attention_ques[:, ques_len]
        high_attention_para = torch.cat([high_attention_ques[:, :ques_len], high_attention_context], 1)

        final_fusion_lstm_input = torch.cat([lstm_ouput, low_attention_para, middle_attention_para, high_attention_para], 2)

        HA, _ = self.final_fusion_lstm(final_fusion_lstm_input, para_mask, ignore_length)
        A = torch.cat([input, HA], 2) # concat input and final fusion

        A_attention = self.self_attention(A, para_mask, A)

        last_lstm_input = torch.cat([HA, A_attention], 2)
        last_lstm_output, _ = self.last_bilstm(last_lstm_input, para_mask, ignore_length)

        output_ques = last_lstm_output[:, :ques_len]
        output_context = last_lstm_output[:, ques_len:]

        return output_ques, output_context, ques_mask, context_mask

    def decode_forward(self, data, output_ques, output_context, ques_mask, context_mask):
        """
        output_ques: batch_size, m, vec_size
        output_context: batch_size, n+1, vec_size
        """
        # vec_size = output_ques.size()[2]
        c_q_logits = self.ques_linear(output_ques).squeeze(2) #batch_size, m
        _, c_q_probs = masked_softmax(c_q_logits, ques_mask, 1) # batch_size, m
        c_q = torch.bmm(c_q_probs.unsqueeze(1), output_ques).squeeze(1) # batch_size, vec_size

        true_start_prob_dist, true_end_prob_dist, true_start_logits, true_end_logits = self.true_answer_pointer(c_q, output_context, context_mask) # batch_size, n+1
        fake_start_prob_dist, fake_end_prob_dist, fake_start_logits, fake_end_logits = self.fake_answer_pointer(c_q, output_context, context_mask)

        ## answer verifier
        cs = torch.bmm(true_start_prob_dist.unsqueeze(1), output_context).squeeze(1) # batch_size, vec_size
        ce = torch.bmm(true_end_prob_dist.unsqueeze(1), output_context).squeeze(1)
        unify_node = output_context[:, 0]
        f = torch.cat([c_q, unify_node, cs, ce], 1) # batch_size, 4*vec_size
        answer_logit = self.answer_verifier_linear(f) # batch_size, 2
        # answer_prob = torch.sigmoid(answer_logit)

        return true_start_prob_dist, true_end_prob_dist, fake_start_prob_dist, fake_end_prob_dist, answer_logit, true_start_logits, true_end_logits, fake_start_logits, fake_end_logits



    def compute_loss(self, data, answer_logit, true_start_logits, true_end_logits, fake_start_logits, fake_end_logits):
        """
            "true_starts" : true_starts, (shape batch_size)
            "true_ends" : true_ends,
            "fake_starts" : fake_starts,
            "fake_ends" : fake_ends,
            "has_ans": has_ans
        """
        # ## has ans, use gather to get prob
        # # true_starts = data["true_starts"].unsqueeze(1)
        # # print(true_starts)
        # true_starts_logits = torch.gather(true_start_prob_dist, 1, true_starts).squeeze(1) # batch_size
        # # print(true_starts_logits)
        # true_ends = data["true_ends"].unsqueeze(1)
        # true_ends_logits = torch.gather(true_end_prob_dist, 1, true_ends).squeeze(1) # batch_size

        # has_ans = data["has_ans"]
        # has_ans = has_ans.float()
        # loss_a = - (torch.mean(torch.log(true_starts_logits)*has_ans) + torch.mean(torch.log(true_ends_logits)*has_ans))

        # ## no ans
        # no_answer = torch.zeros_like(true_starts).long().to(device)
        # fake_starts = data["fake_starts"].unsqueeze(1)
        # fake_ends = data["fake_ends"].unsqueeze(1)
        # zero_starts_logits = torch.gather(true_start_prob_dist, 1, no_answer).squeeze(1) # batch_size
        # zero_ends_logits = torch.gather(true_end_prob_dist, 1, no_answer).squeeze(1) # batch_size
        
        # loss_na = - (torch.mean((1.0-has_ans)*torch.log(zero_starts_logits)) + torch.mean((1.0-has_ans)*torch.log(zero_ends_logits)))

        # fake_starts_logits = torch.gather(fake_start_prob_dist, 1, fake_starts).squeeze(1)
        # fake_ends_logits = torch.gather(fake_end_prob_dist, 1, fake_ends).squeeze(1)

        # loss_na += -(torch.mean((1.0-has_ans)*torch.log(fake_starts_logits)) + torch.mean((1.0-has_ans)*torch.log(fake_ends_logits)))
        has_ans = data["has_ans"]
        true_starts = data["true_starts"]
        true_ends = data["true_ends"]
        fake_starts = data["fake_starts"]
        fake_ends = data["fake_ends"]
        loss_a = F.cross_entropy(true_start_logits, true_starts) + F.cross_entropy(true_end_logits, true_ends)
        loss_na = F.cross_entropy(fake_start_logits, fake_starts) + F.cross_entropy(fake_end_logits, fake_ends)
        ## answer verifier
        # has_ans = has_ans.long()
        loss_av = F.cross_entropy(answer_logit, has_ans)

        loss = loss_a+loss_na+loss_av
        return loss
    
    def forward(self, data):
        output_ques, output_context, ques_mask, context_mask = self.encode_forward(data)
        true_start_prob_dist, true_end_prob_dist, fake_start_prob_dist, fake_end_prob_dist, answer_logit, true_start_logits, true_end_logits, fake_start_logits, fake_end_logits = self.decode_forward(data, output_ques, output_context, ques_mask, context_mask)
        loss = self.compute_loss(data, answer_logit, true_start_logits, true_end_logits, fake_start_logits, fake_end_logits)
        del output_ques, output_context, ques_mask, context_mask, true_start_prob_dist, true_end_prob_dist, fake_start_prob_dist, fake_end_prob_dist, answer_logit
        return loss
    
    def prediction(self, data, debug=False):
        output_ques, output_context, ques_mask, context_mask = self.encode_forward(data)
        true_start_prob_dist, true_end_prob_dist, fake_start_prob_dist, fake_end_prob_dist, answer_logit, true_start_logits, true_end_logits, fake_start_logits, fake_end_logits = self.decode_forward(data, output_ques, output_context, ques_mask, context_mask)
        answer_prob = torch.sigmoid(answer_logit[:, 0])
        if debug:
            print(true_start_prob_dist)
            print(true_end_prob_dist)
            print(answer_prob)
        # pred_has_ans = (answer_prob >= 0.7).long()

        # true_start_has_ans, true_end_has_ans = (true_start_prob_dist[:, 1:].argmax(1)+1)*pred_has_ans, (true_end_prob_dist[:, 1:].argmax(1)+1)*pred_has_ans # batch_size
        # fake_start, fake_end = fake_start_prob_dist.argmax(1), fake_end_prob_dist.argmax(1)
        # true_start = torch.zeros_like(true_start_has_ans) + true_start_has_ans
        # true_end = torch.zeros_like(true_start_has_ans) + true_end_has_ans
        true_start = true_start_prob_dist.argmax(1)
        true_end = true_end_prob_dist.argmax(1)
        if debug:
            print(true_start)
            print(true_end)
        return true_start, true_end, answer_prob
    