import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import RegionalEmbedding, Conv, ResBlock

class DPCNN(nn.Module):
    def __init__(self, hidden_size, feature_map, seq_len, num_class, vocab_size, drop_rate):
        super(DPCNN, self).__init__()
        self.hidden_size = hidden_size
        self.feature_map = feature_map
        self.seq_len = seq_len
        self.num_class = num_class
        self.vocab_size = vocab_size
        self.drop_rate = drop_rate

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.regional_embedding = RegionalEmbedding(self.hidden_size, self.feature_map)
        self.conv = Conv(self.feature_map)

        self.resnet = nn.ModuleList()
        while seq_len > 1:
            self.resnet.append(ResBlock(self.feature_map))
            seq_len = seq_len // 2
        print(len(self.resnet))

        self.linear = nn.Linear(self.feature_map, self.num_class)
    
    def forward(self, x):
        """
        x: batch, seq_len
        """
        x_emb = self.embedding(x) # batch, seq_len, hidden_size
        input = x_emb.transpose(1,2) # batch, hidden_size, seq_len
        regional_res = self.regional_embedding(input) # batch, feature_map, seq_len
        conv_res = self.conv(regional_res) # batch, feature_map, seq_len
        for resnet in self.resnet:
            conv_res = resnet(conv_res)
        assert conv_res.size(2) == 1
        final_input = F.dropout(conv_res.squeeze(2), self.drop_rate, self.training) # batch, feature_map
        logits = self.linear(final_input) # batch, num_class
        return logits
    
    def compute_loss(self, logits, target):
        """
        logits: batch, num_class
        target: batch
        """
        return F.cross_entropy(logits, target)
    
    def compute_res(self, logits):
        """
        logits: batch, num_class
        """
        return torch.argmax(logits, 1)


        
