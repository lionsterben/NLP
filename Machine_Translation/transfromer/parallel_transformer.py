import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda:0")


class BasicTransformer(nn.Module):
    def __init__(self, key_hidden_size, value_hidden_size):
        super(BasicTransformer, self).__init__()
        # self.query_len = query_len
        # self.context_len = context_len
        self.key_hidden_size = key_hidden_size
        self.value_hidden_size = value_hidden_size

    def forward(self, query, key, value, mask = None):
        """
        query: batch, (head), query_len, key_hidden_size
        key  : batch, (head), context_len, key_hidden_size
        value: batch, (head), context_len, value_hidden_size
        mask:  batch, (head), query_len, context_len (1 is real value, 0 is mask)
        """
        atten_logits = torch.matmul(query, key.transpose(-2, -1)) # batch, (head), query_len, context_len
        atten_logits_scale = atten_logits / math.sqrt(self.key_hidden_size)
        if mask is not None:
            atten_logits_scale = (1.0-mask)*(-1e30) + atten_logits_scale
        atten_probs = F.softmax(atten_logits_scale, -1) # batch, (heads), query_len, context_len
        attention = torch.matmul(atten_probs, value) # batch, (heads), query_len, value_hidden_size
        return attention


class Multi_HeadAttention(nn.Module):
    def __init__(self, d_model, key_hidden_size, value_hidden_size, heads):
        super(Multi_HeadAttention, self).__init__()
        self.d_model = d_model
        self.key_hidden_size = key_hidden_size
        self.value_hidden_size = value_hidden_size
        self.heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_model, bias=False) for _ in range(3)])
        self.attention = BasicTransformer(key_hidden_size, value_hidden_size)
        self.output = nn.Linear(self.heads*self.value_hidden_size, self.d_model)
    
    def forward(self, origin_query, origin_key, origin_value, mask = None):
        """
        query: batch, query_len, d_model
        key  : batch, context_len, d_model
        value: batch, context_len, d_model
        mask:  batch, query_len, context_len (1 is real value, 0 is mask)
        """
        batch_size = origin_query.size(0)
        query, key, value = [linear_layer(x).view(batch_size, -1, self.heads, self.key_hidden_size).transpose(1,2) \
                             for linear_layer, x in zip(self.linear_layers, (origin_query, origin_key, origin_value))] # batch, heads, seq_len, d_k
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1)
        multi_head = self.attention(query, key, value, mask).transpose(1, 2).contiguous().view(batch_size, -1, self.heads*self.key_hidden_size)

        output = self.output(multi_head) # batch, query_len, d_model
        return output


        




        

