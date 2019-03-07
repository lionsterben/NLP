import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from layer import Multi_HeadAttention, Feed_Forward
from util import position_embedding, predict_position_embedding

device = torch.device("cuda:0")


class Transformer(nn.Module):
    """
    this model is from attention is all need, encoder and decoder both just have one layer
    """
    def __init__(self, source_vocab_size, target_vocab_size, d_model, key_hidden_size, value_hidden_size, heads, ff_size, drop_rate=0):
        super(Transformer, self).__init__()
        self.source_embedding = nn.Embedding(source_vocab_size, d_model, 0)
        self.target_embedding = nn.Embedding(target_vocab_size, d_model, 0)

        self.encoder_transformer = Multi_HeadAttention(d_model, key_hidden_size, value_hidden_size, heads)
        self.decoder_mask_transformer = Multi_HeadAttention(d_model, key_hidden_size, value_hidden_size, heads)
        self.decoder_transformer = Multi_HeadAttention(d_model, key_hidden_size, value_hidden_size, heads)

        self.encoder_forward = Feed_Forward(d_model, ff_size)
        self.decoder_forward = Feed_Forward(d_model, ff_size)

        self.d_model = d_model
        self.key_hidden_size = key_hidden_size
        self.value_hidden_size = value_hidden_size
        self.heads = heads
        self.drop_rate = drop_rate
        self.layer_norm = nn.LayerNorm(d_model)

        self.d_model2vocab = nn.Linear(d_model, target_vocab_size, bias=False)

        loss_weight_mask = torch.ones(target_vocab_size).to(device)
        loss_weight_mask[0] = 0
        self.criterion = nn.CrossEntropyLoss(weight=loss_weight_mask)
    
    def add_norm(self, input, origin_input):
        """
        input, origin_input: batch, seq_len, d_model
        use dropout. add. layernorm
        """
        dropout = F.dropout(input, self.drop_rate, self.training)
        res_temp = dropout + origin_input
        res = self.layer_norm(res_temp)
        return res

    def encode_forward(self, input):
        """
        input :batch, seq_len
        key, value, query: input
        """
        batch_size, seq_len = input.size(0), input.size(1)
        positional_encoding = torch.tensor(position_embedding(seq_len, self.d_model)).unsqueeze(0).repeat(batch_size, 1, 1).float().to(device)  # batch, seq_len, d_model
        # print(positional_encoding.dtype)
        input_emb = self.source_embedding(input)  # bacth, seq_len, d_model
        # print(input_emb.dtype)
        input_transformer = F.dropout(input_emb + positional_encoding, self.drop_rate, self.training)  # paper say do this
        transformer_output = self.encoder_transformer(input_transformer, input_transformer, input_transformer)  # batch, seq_len, d_model
        # transformer_output_dropout = F.dropout(transformer_output, self.drop_rate, self.training) # paper says do this
        # residual_output = transformer_output_dropout + input_transformer # residual operation
        # layernorm_output = F.layer_norm(residual_output, self.d_model) # apply layer norm in last dimension
        transformer_norm_output = self.add_norm(transformer_output, input_transformer)
        feed_forward = self.encoder_forward(transformer_norm_output) # batch, seq_len, d_model
        # output = feed_forward + layernorm_output
        # output_norm = F.layer_norm(output, self.d_model) # batch, seq_len, d_model
        output = self.add_norm(feed_forward, transformer_norm_output)
        return output
    
    def decode_forward(self, output, encoder_output):
        batch_size, seq_len = output.size(0), output.size(1)
        positional_encoding = torch.tensor(position_embedding(seq_len, self.d_model)).unsqueeze(0).repeat(batch_size, 1, 1).float().to(device) # batch, seq_len, d_model
        # positional_encoding = torch.tensor(predict_position_embedding(pos, self.d_model)).unsqueeze(0).repeat(batch_size, 1, 1).float().to(device) # batch, 1, d_model
        output_emb = self.target_embedding(output)
        mask_transformer_input = F.dropout(positional_encoding + output_emb, self.drop_rate, self.training)
        # mask is query_len, context_len
        mask = torch.zeros(batch_size, seq_len, seq_len).to(device) # 屏蔽掉预测当前单词时候能看到input之后的单词
        for idx in range(seq_len):
            mask[:, idx, 0:idx+1] = 1.0
        mask_transformer_output = self.decoder_mask_transformer(mask_transformer_input, mask_transformer_input,
                                                               mask_transformer_input, mask=mask)
        # mask_residual_output = mask_residual_output + mask_transformer_input
        # mask_norm = F.layer_norm(mask_residual_output, self.d_model)
        mask_output = self.add_norm(mask_transformer_output, mask_transformer_input)

        transformer_output = self.decoder_transformer(mask_output, encoder_output, encoder_output)
        output = self.add_norm(transformer_output, mask_output)
        
        feed_forward = self.decoder_forward(output)
        final_output = self.add_norm(feed_forward, output)
        output_logits = self.d_model2vocab(final_output)  # batch, seq_len, vocab_size
        return output_logits

    def forward(self, input, output):
        """
        input is encoder input
        output is decoder input
        """
        encoder_output = self.encode_forward(input)
        decoder_output = self.decode_forward(output, encoder_output)

        return decoder_output

    def compute_loss(self, logits, target):
        loss = self.criterion(logits.view(-1, logits.size()[-1]), target.view(-1))
        return loss







        
        


