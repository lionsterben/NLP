import torch
import torch.nn as nn

import sys
sys.path.append("/home/FuDawei/NLP")

from torch import optim
import torch.nn.functional as F

device = torch.device("cuda:0")



class Basic_NMT(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, source_emb_size, target_emb_size, source_hidden_size, target_hidden_size, drop_rate = 0):
        super(Basic_NMT, self).__init__()
        self.source_emb = nn.Embedding(source_vocab_size, source_emb_size, 0)
        self.target_emb = nn.Embedding(target_vocab_size, target_emb_size, 0)
        self.source_hidden_size = source_hidden_size // 2
        self.encoder = nn.LSTM(source_emb_size, self.source_hidden_size, 2, batch_first=True, dropout = drop_rate, bidirectional = True)
        self.decoder = nn.LSTM(target_emb_size, target_hidden_size, 1, batch_first=True, dropout = drop_rate, bidirectional = False)
        self.transfrom_hidden = nn.Linear(self.source_hidden_size*2, target_hidden_size, bias = False)
        self.transfrom_cell = nn.Linear(self.source_hidden_size*2, target_hidden_size, bias = False)
        # self.init_parameters()
        self.target_hidden_size = target_hidden_size
        self.target2vocab = nn.Linear(target_hidden_size, target_vocab_size)
        loss_weight_mask = torch.ones(target_vocab_size).to(device)
        loss_weight_mask[0] = 0
        self.criterion = nn.CrossEntropyLoss(weight = loss_weight_mask)
    
    def init_parameters(self):
        # nn.init.normal_(self.encoder., -0.08, 0.08)
        # nn.init.normal_(self.decoder.parameters(), -0.08, 0.08)
        def init_lstm(m):
            for param in m.parameters():
                if len(param.size()) >=2 :
                    nn.init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        self.source_emb.weight.data.uniform_(-0.1, 0.1)
        self.target_emb.weight.data.uniform_(-0.1, 0.1)
        init_lstm(self.encoder)
        init_lstm(self.decoder)

    def encode(self, encoder_input, encoder_input_mask):
        encoder_input_emb = self.source_emb(encoder_input)
        _, (hidden_state, cell_state) = self.encoder(encoder_input_emb)
        h_t = torch.cat((hidden_state[-1], hidden_state[-2]), 1)
        c_t = torch.cat((cell_state[-1], cell_state[-2]), 1)

        h_t = self.transfrom_hidden(h_t)
        c_t = self.transfrom_cell(c_t)
        h_t = h_t.view(1, h_t.size(0), h_t.size(1))
        c_t = c_t.view(1, c_t.size(0), c_t.size(1))
        return (h_t, c_t)
    
    def decode(self, decoder_input, encoder_state):
        h_t, c_t = encoder_state
        decoder_input_emb = self.target_emb(decoder_input)
        decoder_output, decode_state = self.decoder(decoder_input_emb, encoder_state)
        
        decoder_logits = self.target2vocab(decoder_output) ## batch, seq, vocab_size
        # print(decoder_logits)
        return decoder_logits, decode_state

    
    def forward(self, encoder_input, encoder_input_mask, decoder_input):
        """encoder_input : id"""
        encoder_state = self.encode(encoder_input, encoder_input_mask)
        decoder_logits, _ = self.decode(decoder_input, encoder_state)

        # print(decoder_logits)
        return decoder_logits
    
    def compute_loss(self, logits, target):
        loss = self.criterion(logits.view(-1, logits.size()[-1]), target.view(-1))
        return loss
    
    def compute_prob(self, logits):
        word_prob = F.softmax(logits, 2)
        return word_prob

    
    


