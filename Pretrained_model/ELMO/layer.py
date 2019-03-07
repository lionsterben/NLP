import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    """
    elmo paper say need this
    y = g*x + (1-g)*f(A(x))
    A is linear transformation, f is non-linearity(relu), g from sigmoid(linear_B(x))
    """
    def __init__(self, hidden_size, num_layers):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.linear_A = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.linear_B = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        for idx in range(num_layers):
            self.linear_B[idx].bias.data.fill_(1.0)
    
    def forward(self, x):
        """
        x: batch, unknown, hidden_size
        """
        res = x
        for idx in range(self.num_layers):
            gate = torch.sigmoid(self.linear_B[idx](res))
            res = gate*res + (1.0-gate)*F.relu(self.linear_A[idx](res))
        return res





class Convlayer(nn.Module):
    """
    transform character_id to word represtion
    use 'filters': [[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]]
    input : batch, seq_len, char_len
    output: batch, seq_len, embedding
    this method is basic textcnn from kim
    """
    def __init__(self, char_size, char_emb_size, embedding_size):
        super(Convlayer, self).__init__()
        self.char_embedding = nn.Embedding(char_size, char_emb_size)
        self.filters = [[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]]
        self.conv_layer = nn.ModuleList()
        ## 设置conv layer
        for kernel_size, out_channels in self.filters:
            conv = nn.Conv1d(char_emb_size, out_channels, kernel_size)
            self.conv_layer.append(conv)
        filters_num = sum(f[1] for f in self.filters)
        self.highway = Highway(filters_num, 2)
        self.linear = nn.Linear(filters_num, embedding_size)
        self.embedding_size = embedding_size
        

    def forward(self, x):
        """
        x: batch, seq_len, char_len
        char_len is 50
        """
        batch_size = x.size(0)
        # print(x.size)
        # print(torch.sum(torch.max(x.view(-1, 50), -1)[0]>261))
        x_emb = self.char_embedding(x.view(-1, 50)).transpose(1, 2) # batch*seq_len, char_emb_size, char_len
        conv_final = []
        for conv in self.conv_layer:
            conv_res = conv(x_emb) # batch*seq_len, out_channel, unknown
            conv_res, _ = torch.max(conv_res, -1) # batch*seq_len, out_channel
            conv_final.append(conv_res)
        res = torch.cat(conv_final, -1) # batch*seq_len, sum of out_channel
        highway_res = self.highway(res)
        res = self.linear(highway_res).view(batch_size, -1, self.embedding_size) # batch, seq_len, embedding_size
        return res

class MyLstm(nn.Module):
    """
    this module is for forward and baward lstm in elmo
    for example: [<s>, I, love, you, </s>]. 
    In forward, input is [<s>, I, love, you], ground_truth is [I, love, you, </s>]
    In backward, input is [</s>, you, love, I]， ground_truth is [you, love, I, <s>]
    input: batch, seq_len, embedding_size
    mask: batch, seq_len 1 is real, 0 is padding
    """
    def __init__(self, input_size, hidden_size, drop_rate):
        super(MyLstm, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(hidden_size, input_size) 
        self.drop_rate = drop_rate


    def forward(self, input, mask):
        batch_size = input.size()[0]
        # context_mask = masks[:, ignore_length:]
        length = mask.sum(1) # batch_size
        sort_length, indices = length.sort(descending=True)
        sort_input = input.index_select(0, indices) # descend for length
        x = nn.utils.rnn.pack_padded_sequence(sort_input, sort_length, batch_first=True)
        output, _ = self.lstm(x)
        output, length = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        back_indices = indices.argsort()
        output = output.index_select(0, back_indices) # batch, seq_len, hidden_size
        output = self.linear(output)
        output = F.dropout(output, self.drop_rate, self.training) # batch, seq_len, input_size
        return output







