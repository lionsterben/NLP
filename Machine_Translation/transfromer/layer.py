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
        query: batch, query_len, key_hidden_size
        key  : batch, context_len, key_hidden_size
        value: batch, context_len, value_hidden_size
        mask:  batch, query_len, context_len (1 is real value, 0 is mask)
        """
        atten_logits = torch.bmm(query, key.transpose(1, 2)) # batch, query_len, context_len
        atten_logits_scale = atten_logits / math.sqrt(self.key_hidden_size)
        if mask is not None:
            atten_logits_scale = (1.0-mask)*(-1e30) + atten_logits_scale
        atten_probs = F.softmax(atten_logits_scale, 2) # batch, query_len, context_len
        attention = torch.bmm(atten_probs, value) # batch, query_len, value_hidden_size
        return attention


class Multi_HeadAttention(nn.Module):
    def __init__(self, d_model, key_hidden_size, value_hidden_size, heads):
        super(Multi_HeadAttention, self).__init__()
        self.d_model = d_model
        self.key_hidden_size = key_hidden_size
        self.value_hidden_size = value_hidden_size
        self.heads = heads
        self.projection = []
        for _ in range(self.heads):
            project_q = nn.Linear(self.d_model, self.key_hidden_size, bias=False).to(device)
            project_k = nn.Linear(self.d_model, self.key_hidden_size, bias=False).to(device)
            project_v = nn.Linear(self.d_model, self.value_hidden_size, bias=False).to(device)
            self.projection.append((project_q, project_k, project_v))
        self.attention = BasicTransformer(key_hidden_size, value_hidden_size)
        self.output = nn.Linear(self.heads*self.value_hidden_size, self.d_model)
    
    def forward(self, origin_query, origin_key, origin_value, mask = None):
        """
        query: batch, query_len, d_model
        key  : batch, context_len, d_model
        value: batch, context_len, d_model
        mask:  batch, query_len, context_len (1 is real value, 0 is mask)
        """
        head = []
        for (project_q, project_k, project_v) in self.projection:
            query = project_q(origin_query) # batch, query_len, key_hidden_size
            key = project_k(origin_key) # batch, context_len, key_hidden_size
            value = project_v(origin_value) # batch, context_len, value_hidden_size
            head.append(self.attention(query, key, value, mask))
        multi_head = None
        for i in head:
            if multi_head is None:
                multi_head = i
            else:
                multi_head = torch.cat([multi_head, i], 2) # batch, query_len, value_hidden_size*h
        output = self.output(multi_head) # batch, query_len, d_model
        return output

class Feed_Forward(nn.Module):
    def __init__(self, d_model, hidden_size):
        super(Feed_Forward, self).__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.d_model2hidden = nn.Linear(d_model, hidden_size)
        self.hidden2d_model = nn.Linear(hidden_size, d_model)
    
    def forward(self, input):
        """
        input: batch, query_len, d_model
        """
        hidden = F.relu(self.d_model2hidden(input)) # batch, query_len, hidden_size
        output = self.hidden2d_model(hidden) # batch, query_len, d_model
        return output