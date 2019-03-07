"""this file is input bilstm and attention module"""

import tensorflow as tf

class LSTMEncoder():
    def __init__(self, hidden_size, keep_prob):
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        self.lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(self.lstm_cell_fw, input_keep_prob = self.keep_prob)
        self.lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        self.lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(self.lstm_cell_bw, input_keep_prob = self.keep_prob)

    def build_graph(self, context, context_mask, question, question_mask, question_len):
        """
        unify node is key node to merge information of question and context, however question and context have padding mask, 
        for convenient, I just delete context mask.
        unify node: shape (batch_size, embedding_size)
        """
        with tf.variable_scope("LSTMEncoder"):
            inputs = tf.concat([question, context], axis=1)
            input_lens = tf.reduce_sum(context_mask, 1) + tf.tile(tf.constant(question_len, shape = [1]), [tf.shape(context)[0]])
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.lstm_cell_fw, self.lstm_cell_bw, inputs, input_lens, dtype=tf.float32)
            out = tf.concat([fw_out, bw_out], 2)
            out = tf.nn.dropout(out, self.keep_prob)
            return out

class BasicAttn():
    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size
    
    def build_graph(self, values, values_mask, keys):
        """ focus on key, every key has distribution on values"""
        with tf.variable_scope("BasicAtten"):
            values_t = tf.transpose(values, perm=[0,2,1])
            attn_logits = tf.matmul(keys, values_t)
            attn_logits_mask = tf.expand_dims(values_mask, 1)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2)
            output = tf.matmul(attn_dist, values) #shape (batch_size, num_keys, value_vec_size)
            output = tf.nn.dropout(output, self.keep_prob)
            return attn_dist, output


def masked_softmax(logits, mask, dim):
    exp_mask = (1-tf.cast(mask, 'float32'))*(-1e30)
    masked_logits = tf.add(logits, exp_mask)
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist

        

        
        

        
