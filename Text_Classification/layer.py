import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionalEmbedding(nn.Module):
    def __init__(self, hidden_size, feature_map):
        super(RegionalEmbedding, self).__init__()
        self.hidden_size = hidden_size
        self.feature_map = feature_map
        self.batch_norm = nn.BatchNorm1d(self.feature_map)
        self.region_1 = nn.Conv1d(self.hidden_size, self.feature_map, kernel_size=1) # batch, feature_map, seq_len
        self.region_3 = nn.Conv1d(self.hidden_size, self.feature_map, kernel_size=3, padding=1) # batch, feature_map, seq_len
        self.region_5 = nn.Conv1d(self.hidden_size, self.feature_map, kernel_size=5, padding=2) # batch, feature_map, seq_len 

    def forward(self, x):
        """
        x: batch, hidden_size, seq_len
        """

        region_embedding = self.region_1(x) + self.region_3(x) + self.region_5(x)
        # region_embedding = self.region_3(x)
        batch_norm = self.batch_norm(region_embedding)
        
        return batch_norm

class RegionalEmbedding2(nn.Module):
    def __init__(self, hidden_size, feature_map):
        super(RegionalEmbedding, self).__init__()
        self.hidden_size = hidden_size
        self.feature_map = feature_map
        self.batch_norm = nn.BatchNorm1d(self.feature_map)
        # self.region_1 = nn.Conv1d(self.hidden_size, self.feature_map, kernel_size=1) # batch, feature_map, seq_len
        self.region_3 = nn.Conv1d(self.hidden_size, self.feature_map, kernel_size=3, padding=1) # batch, feature_map, seq_len
        # self.region_5 = nn.Conv1d(self.hidden_size, self.feature_map, kernel_size=5, padding=2) # batch, feature_map, seq_len 

    def forward(self, x):
        """
        x: batch, hidden_size, seq_len
        """

        # region_embedding = self.region_1(x) + self.region_3(x) + self.region_5(x)
        region_embedding = self.region_3(x)
        batch_norm = self.batch_norm(region_embedding)
        output = F.dropout(batch_norm, 0.2, self.training)
        
        return output

class Conv(nn.Module):
    def __init__(self, feature_map):
        super(Conv, self).__init__()
        self.feature_map = feature_map
        self.conv1 = nn.Conv1d(self.feature_map, self.feature_map, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv1d(self.feature_map, self.feature_map, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(self.feature_map)
        self.batch_norm2 = nn.BatchNorm1d(self.feature_map)
    
    def forward(self, x):
        """
        x: batch, feature_map, seq_len
        """
        res_1 = self.batch_norm1(self.conv1(F.relu(x)))
        res_2 = self.batch_norm2(self.conv2(F.relu(res_1)))
        return x + res_2

class ResBlock(nn.Module):
    def __init__(self, feature_map):
        """
        x: batch, feature_map, seq_len(/2)
        x need pad first
        """
        super(ResBlock, self).__init__()
        self.feature_map = feature_map
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv1 = nn.Conv1d(self.feature_map, self.feature_map, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv1d(self.feature_map, self.feature_map, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(self.feature_map)
        self.batch_norm2 = nn.BatchNorm1d(self.feature_map)
    
    def forward(self, x):
        pad_x = F.pad(x, (0, 1))
        pool_x = self.max_pool(pad_x)
        res_1 =  self.batch_norm1(self.conv1(F.relu(pool_x)))
        res_2 = self.batch_norm2(self.conv2(F.relu(res_1)))
        
        return pool_x + res_2



