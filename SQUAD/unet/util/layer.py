## bilstm(p)
## full attention(p,q)
## self attention(p)

## use official implemention
# class Bilstm(nn.module):
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
def masked_softmax(logits, mask, dim):
    """mask: torch tensor 1 for real, 0 for pad. shape (batch_size, num_values)"""
    exp_mask = ((1-mask)*(-1e30)).float()
    masked_logits = logits + exp_mask
    prob_dist = F.softmax(masked_logits, dim)
    return masked_logits, prob_dist



class BasicAttention(nn.Module):
    def __init__(self, drop_rate, key_vec_size, value_vec_size):
        super(BasicAttention, self).__init__()
        self.drop_rate = drop_rate
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size
        self.linear = nn.Linear(key_vec_size, value_vec_size, bias=False)
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.xavier_normal_(self.linear.weight.data)

    def forward(self, values, values_mask, keys):
        """
        values: (batch_size, num_values, value_vec_size)
        values_mask: (batch_size, num_values)
        keys: (batch_size, num_keys, key_vec_size)
        """
        values_t = values.transpose(1,2) # (batch_size, value_vec_size, num_values)
        attn_logits = torch.bmm(self.linear(keys), values_t) # (batch_size, num_keys, num_values)
        attn_logits_mask = values_mask.unsqueeze(1) # (batch_size, 1, num_values)
        _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # (batch_size, num_keys, num_values)
        # output = attn_dist.matmul(values) # (batch_size, num_keys, value_vec_size)
        output = torch.bmm(attn_dist, values)
        output = F.dropout(output, self.drop_rate, self.training)
        return output

class BiAttention(nn.Module):
    ## do support mask
    def __init__(self, drop_rate, vec_size, attention_size):
        super(BiAttention, self).__init__()
        self.drop_rate = drop_rate
        self.vec_size = vec_size
        self.attention_size = attention_size
        self.linear1 = nn.Linear(vec_size, attention_size, bias=False)
        self.linear2 = nn.Linear(vec_size, attention_size, bias=False)
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.xavier_normal_(self.linear1.weight.data)
        nn.init.xavier_normal_(self.linear2.weight.data)

    
    def forward(self, q, p, q_mask, p_mask):
        """
        q: (batch_size, m+1, vec_size)
        p: (batch_size, n+1, vec_size)
        q_mask: (batch_size, m+1) 1 for real
        p_mask: (batch_size, n+1) 0 for pad
        """
        ### this layer has some bug, for key mask is 0 ,all weights is -1e30, so value weights is equal
        s_q = F.relu(self.linear1(q)) # (batch_size, m+1, attention_size)
        s_p = F.relu(self.linear2(p)).transpose(1,2) # (batch_size, attention_size, n+1)
        s = torch.bmm(s_q, s_p) # (batch_size, m+1, n+1)
        # q_mask, p_mask = q_mask.float(), p_mask.float()
        # mask = torch.bmm(q_mask.unsqueeze(2), p_mask.unsqueeze(1)) # (batch_size, m+1, n+1)
        # exp_mask = ((1-mask)*(-1e30)).float()
        # s_mask = s + exp_mask # (batch_size, m+1, n+1)
        # s_mask_for_q = F.softmax(s_mask, 2)
        # output_q = torch.bmm(s_mask_for_q, p) # (batch_size, m+1, vec_size)
        # s_mask_for_p = F.softmax(s_mask.transpose(1,2), 2) # (batch_size, n+1, m+1)
        # output_p = torch.bmm(s_mask_for_p, q) # (batch_size, n+1, vec_size)

        # for use masked fill, we inverse mask flag, 1 is mask(pad), 0 is real
        q_mask, p_mask = (1-q_mask).byte(), (1-p_mask).byte()
        s_p = s.clone()
        mask_p = p_mask.unsqueeze(1).repeat(1, q.size(1), 1)
        s_p.data.masked_fill_(mask_p.data, -float('inf'))
        weights_p = F.softmax(s_p, 2)
        output_q = torch.bmm(weights_p, p)

        s = s.transpose(1,2)
        mask_q = q_mask.unsqueeze(1).repeat(1, p.size(1), 1)
        s.data.masked_fill_(mask_q.data, -float('inf'))
        weights_q = F.softmax(s, 2)
        output_p = torch.bmm(weights_q, q)


        output_p = F.dropout(output_p, self.drop_rate, self.training)
        output_q = F.dropout(output_q, self.drop_rate, self.training)

        return output_p, output_q



class Bilstm(nn.Module):
    """need ignore padding 0"""
    def __init__(self, input_size, hidden_size, drop_rate):
        super(Bilstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.drop_rate = drop_rate
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
    
    def init_hidden_state(self, batch_size):
        h0 = torch.zeros((2, batch_size, self.hidden_size)).to(device)
        c0 = torch.zeros((2, batch_size, self.hidden_size)).to(device)
        return (nn.init.xavier_normal_(h0), nn.init.xavier_normal_(c0))
    
    def forward(self, inputs, masks, ignore_length):
        """
        inputs: (batch_size, seq_len, input_size)
        masks: (batch_size, seq_len)
        this module is batch first
        ignore_length: ques_max_len
        only ignore context padding, question context padding need cal
        """
        # print(ignore_length)
        # print(masks)
        batch_size = inputs.size()[0]
        context_mask = masks[:, ignore_length:]
        # # print(context_mask.size())
        # # print(context_mask.sum(1))
        length = context_mask.sum(1) + ignore_length # batch_size
        # # print(length.size())
        sort_length, indices = length.sort(descending=True)
        sort_input = inputs.index_select(0, indices) ## descend for length
        x = nn.utils.rnn.pack_padded_sequence(sort_input, sort_length, batch_first=True)
        init_hidden_state = self.init_hidden_state(batch_size)
        output, _ = self.lstm(x, init_hidden_state)
        output, length = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        back_indices = indices.argsort()
        output = output.index_select(0, back_indices)
        output = F.dropout(output, self.drop_rate, self.training)
        return output, 1

class AnswerPoint(nn.Module):
    def __init__(self, ques_vec_size, context_vec_size):
        super(AnswerPoint, self).__init__()
        self.ques_vec_size = ques_vec_size
        self.context_vec_size = context_vec_size
        self.linear_start = nn.Linear(context_vec_size, ques_vec_size, bias=False)
        self.linear_end = nn.Linear(context_vec_size, ques_vec_size, bias=False)
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_normal_(self.linear_start.weight.data)
        nn.init.xavier_normal_(self.linear_end.weight.data)
    
    def forward(self, c_q, p, p_mask):
        """
        c_q: (batch_size, ques_vec_size)
        p: (batch_size, n+1, context_vec_size)
        p_mask: (batch_size, n+1) 
        """
        expand_c_q = c_q.unsqueeze(2) # (batch_size, ques_vec_size, 1)
        start_logits = torch.bmm(self.linear_start(p), expand_c_q).squeeze(2) # (batch_size, n+1)
        end_logits = torch.bmm(self.linear_end(p), expand_c_q).squeeze(2) # (batch_size, n+1)
        start_logits, start_prob_dist = masked_softmax(start_logits, p_mask, 1)
        end_logits, end_prob_dist = masked_softmax(end_logits, p_mask, 1)
        return start_prob_dist, end_prob_dist, start_logits, end_logits