import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import Convlayer, MyLstm

device = torch.device("cuda:1")
# device = torch.device("cpu")

class Elmo(nn.Module):
    """
    input is batch_size sentences which is processed by token2id
    input：batch, seq_len, 50
    mask: batch, seq_len
    input is character level, after convlayer, get word level embedding
    then use lstm to build
    """
    def __init__(self, char_size, char_emb_size, embedding_size, hidden_size, vocab_size, drop_rate):
        super(Elmo, self).__init__()
        self.conv_layer = Convlayer(char_size, char_emb_size, embedding_size)
        # self.conv_layer_backward = Convlayer(char_size, char_emb_size, embedding_size)

        self.forward_lstm1 = MyLstm(embedding_size, hidden_size, drop_rate)
        self.backward_lstm1 = MyLstm(embedding_size, hidden_size, drop_rate)

        self.forward_lstm2 = MyLstm(embedding_size, hidden_size, drop_rate)
        self.backward_lstm2 = MyLstm(embedding_size, hidden_size, drop_rate)


        self.linear_forward = nn.Linear(embedding_size, vocab_size)
        self.linear_backward = nn.Linear(embedding_size, vocab_size)

        self.vocab_size = vocab_size
        loss_weight_mask = torch.ones(vocab_size).to(device)
        loss_weight_mask[0] = 0
        self.criterion = nn.CrossEntropyLoss(weight = loss_weight_mask)
    
    def forward(self, forward_input, forward_mask, backward_input, backward_mask):
        forward_conv = self.conv_layer(forward_input) # batch, seq_len, embedding_size
        backward_conv = self.conv_layer(backward_input) # batch, seq_len, embedding_size

        forward_layer1 = self.forward_lstm1(forward_conv, forward_mask) # batch, seq_len, embedding_size
        backward_layer1 = self.backward_lstm1(backward_conv, backward_mask)

        forward_input_layer2 = forward_layer1 + forward_conv # residual
        backward_input_layer2 = backward_layer1 + backward_conv
        
        forward_layer2 = self.forward_lstm2(forward_input_layer2, forward_mask) # batch, seq_len, embedding_size
        backward_layer2 = self.backward_lstm2(backward_input_layer2, backward_mask)

        forward_output = self.linear_forward(forward_layer2)
        backward_output = self.linear_backward(backward_layer2)

        return forward_output, backward_output
    
    def compute_loss(self, forward_output, backward_output, forward_ground, backward_ground):
        """
        output: batch, seq_len, vocab_size
        ground: batch, seq_len
        """
        # print(forward_output)
        # print(forward_ground)
        loss = self.criterion(forward_output.view(-1, self.vocab_size), forward_ground.view(-1)) + \
                self.criterion(backward_output.view(-1, self.vocab_size), backward_ground.view(-1))
        return loss
    
    def build_represention(self, input, mask):
        """
        input is batch, seq_len, 50. but it's a complete sentence
        """
        conv = self.conv_layer(input) # batch, seq_len, embedding_size
        seq_len = input.size(1)
        order = torch.tensor(list(reversed([i for i in range(seq_len)]))).long()
        reverse_conv = self.conv_layer(torch.index_select(input, 1, order)) # batch, seq_len, embedding_size

        forward_layer1 = self.forward_lstm1(conv, mask) # batch, seq_len, embedding_size
        backward_layer1 = self.backward_lstm1(reverse_conv, mask)

        forward_input_layer2 = forward_layer1 + conv # residual
        backward_input_layer2 = backward_layer1 + reverse_conv
        
        forward_layer2 = self.forward_lstm2(forward_input_layer2, mask) # batch, seq_len, embedding_size//2
        backward_layer2 = self.backward_lstm2(backward_input_layer2, mask)

        

        represention = {"layer_1": torch.cat[conv, torch.index_select(reverse_conv, 1, order), 2], "layer_2": torch.cat[forward_layer1, torch.index_select(backward_layer1, 1, order), 2],\
                        "layer_3": torch.cat[forward_layer2, torch.index_select(backward_layer2, 1, order), 2]} # every label is batch, seq_len, 2*embedding_size

        return represention
        

