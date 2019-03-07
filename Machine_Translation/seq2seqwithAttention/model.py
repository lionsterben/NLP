import torch
import torch.nn as nn

import sys
sys.path.append("/home/FuDawei/NLP")

from torch import optim
import torch.nn.functional as F

device = torch.device("cuda:1")



class Attention_NMT(nn.Module):
    """this model is from Effective Approaches to Attention-based NMT paper, not origin paper from Bahdanau"""
    def __init__(self, source_vocab_size, target_vocab_size, source_emb_size, target_emb_size, source_hidden_size, target_hidden_size, drop_rate = 0):
        super(Attention_NMT, self).__init__()
        self.source_emb = nn.Embedding(source_vocab_size, source_emb_size, 0)
        self.target_emb = nn.Embedding(target_vocab_size, target_emb_size, 0)
        self.source_hidden_size = source_hidden_size // 2
        self.encoder = nn.LSTM(source_emb_size, self.source_hidden_size, 2, batch_first=True, dropout = drop_rate, bidirectional = True)
        self.decoder = nn.LSTM(target_emb_size, target_hidden_size, 1, batch_first=True, dropout = drop_rate, bidirectional = False)
        self.transfrom_hidden = nn.Linear(self.source_hidden_size*2, target_hidden_size, bias = False)
        self.transfrom_cell = nn.Linear(self.source_hidden_size*2, target_hidden_size, bias = False)
        # self.init_parameters()
        self.target_hidden_size = target_hidden_size
        self.target2vocab = nn.Linear(self.target_hidden_size, target_vocab_size)
        loss_weight_mask = torch.ones(target_vocab_size).to(device)
        loss_weight_mask[0] = 0
        self.criterion = nn.CrossEntropyLoss(weight = loss_weight_mask)
        self.attention = nn.Linear(self.target_hidden_size, self.source_hidden_size*2, bias = False)
        self.decoder_attention2hidden = nn.Linear(self.source_hidden_size*2+self.target_hidden_size, self.target_hidden_size, False)
    
    def init_parameters(self):
        # nn.init.normal_(self.encoder., -0.08, 0.08)
        # nn.init.normal_(self.decoder.parameters(), -0.08, 0.08)
        self.source_emb.weight.data.uniform_(-0.1, 0.1)
        self.target_emb.weight.data.uniform_(-0.1, 0.1)

    def encode(self, encoder_input):
        encoder_input_emb = self.source_emb(encoder_input)
        encode_output, (hidden_state, cell_state) = self.encoder(encoder_input_emb) # output: batch, source_len, 2*self.source_hidden_size
        h_t = torch.cat((hidden_state[-1], hidden_state[-2]), 1)
        c_t = torch.cat((cell_state[-1], cell_state[-2]), 1)

        h_t = self.transfrom_hidden(h_t)
        c_t = self.transfrom_cell(c_t)
        h_t = h_t.view(1, h_t.size(0), h_t.size(1))
        c_t = c_t.view(1, c_t.size(0), c_t.size(1))
        return encode_output, (h_t, c_t)
    
    def decode(self, decoder_input, encoder_state, encoder_input_mask, encoder_output):
        """
            encoder_input_mask: batch, source_len
            encoder_output: batch, source_len, source_hidden_size*2
        """
        h_t, c_t = encoder_state
        decoder_input_emb = self.target_emb(decoder_input)
        target_len = decoder_input.size(1) 
        decoder_output, decode_state = self.decoder(decoder_input_emb, encoder_state) ## decoder_output: batch, target_len, self.target_hidden_size
        atten_temp = self.attention(decoder_output) ## batch, target_len, self.source_hidden_size*2
        atten_logits = torch.bmm(atten_temp, encoder_output.transpose(1,2)) ## batch, target_len, source_len
        encoder_mask = encoder_input_mask.unsqueeze(1).repeat(1, target_len, 1).float() ## batch, target_len, source_len
        mask_logits = (1.0-encoder_mask)*(-1e30) + atten_logits ## batch, target_len, source_len
        atten_score = F.softmax(mask_logits, 2) ## batch, target_len, source_len

        context_vector = torch.bmm(atten_score, encoder_output) ## batch, target_len, source_hidden_size*2
        # context_vector_transform = self.decoder_attention2hidden(context_vector) ## batch, target_len, target_hidden_size
        final_hidden_state = torch.tanh(self.decoder_attention2hidden(torch.cat([context_vector, decoder_output], 2))) ## batch, target_len, source_hidden_size*2+self.target_hidden_size --> batch, target_len, self.target_hidden_size


        
        decoder_logits = self.target2vocab(final_hidden_state) ## batch, seq, vocab_size
        # print(decoder_logits)
        return decoder_logits, decode_state

    
    def forward(self, encoder_input, encoder_input_mask, decoder_input):
        """encoder_input : id"""
        encoder_output, encoder_state = self.encode(encoder_input)
        decoder_logits, _ = self.decode(decoder_input, encoder_state, encoder_input_mask, encoder_output)

        # print(decoder_logits)
        return decoder_logits
    
    def compute_loss(self, logits, target):
        loss = self.criterion(logits.view(-1, logits.size()[-1]), target.view(-1))
        return loss
    
    def compute_prob(self, logits):
        word_prob = F.softmax(logits, 2)
        return word_prob