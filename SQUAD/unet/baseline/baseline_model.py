import tensorflow as tf
import numpy as np
from module import LSTMEncoder, BasicAttn
from data_util.batch_data import get_batch_generator
from evaluate import compute_exact, compute_f1

import time
import logging
import os
import sys


logging.basicConfig(level=logging.INFO)

class Baseline():

    def __init__(self, FLAGS, id2word, word2id, emb_matrix):
        print("init the baseline model")
        self.FLAGS = FLAGS
        self.id2word = id2word
        self.word2id = word2id
        self.emb_matrix = emb_matrix
        # self.unify_node = tf.get_variable(name="unifyNode", dtype=tf.float32, shape=(FLAGS.embedding_size), initializer=tf.contrib.layers.xavier_initializer())
        with tf.variable_scope("baseline", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.add_placeholders()
            # self.unify_node = tf.get_variable(name="unifyNode", dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), validate_shape=False)
            self.add_embedding_layer(emb_matrix)
            self.build_graph()
            self.add_loss()
        

        # Define trainable parameters, gradient, gradient norm, and clip by gradient norm
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.param_norm = tf.global_norm(params)

        # Define optimizer and updates
        # (updates is what you need to fetch in session.run to do a gradient update)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) # you can try other optimizers
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        # Define savers (for checkpointing) and summaries (for tensorboard)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
        self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.summaries = tf.summary.merge_all()
    
    def add_placeholders(self):
        """
        Add placeholders to the graph. Placeholders are used to feed in inputs.
        """
        # Add placeholders for inputs.
        # These are all batch-first: the None corresponds to batch_size and
        # allows you to run the same model with variable batch_size
        self.context_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.context_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.qn_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.qn_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.ans_span = tf.placeholder(tf.int32, shape=[None, 2])
        # self.ans_span = self.ans_span + 1 ## context need contain unifyNode in the first
        self.impossible = tf.placeholder(tf.float32, shape=[None])
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())
    
    def add_embedding_layer(self, emb_matrix):
        with tf.variable_scope("embeddings"):
            embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32, name="emb_matrix")
            self.context_embs = tf.nn.embedding_lookup(embedding_matrix, self.context_ids)
            self.qn_embs = tf.nn.embedding_lookup(embedding_matrix, self.qn_ids)
    
    def build_graph(self):
        ## LSTM encoder
        encoder = LSTMEncoder(self.FLAGS.hidden_size, self.keep_prob)
        hidden = encoder.build_graph(self.context_embs, self.context_mask, self.qn_embs, self.qn_mask, self.FLAGS.question_len)
        ## context_hidden :context+und
        qn_hidden, context_hidden = hidden[:, :self.FLAGS.question_len], hidden[:, self.FLAGS.question_len:]
        ## question context attention
        # unify_hidden = tf.expand_dims(unify_hidden, 1)
        # qn_unify = tf.concat([qn_hidden, unify_hidden], 1)
        batch_size = tf.shape(self.context_ids)[0]
        # tf.Print(batch_size)
        temp = tf.constant(1, shape=[1,1])
        qn_unify_mask = tf.concat([self.qn_mask, tf.tile(temp, [batch_size,1])], 1)
        # unify_context = tf.concat([unify_hidden, context_hidden], 1)
        # context_mask = tf.concat([tf.tile(temp, [batch_size,1]), self.context_mask], 1)
        qn2context_attn = BasicAttn(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
        _, qn_attn_output = qn2context_attn.build_graph(context_hidden, self.context_mask, hidden[:,:self.FLAGS.question_len+1])
        context2qn_attn = BasicAttn(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
        _, context_attn_output = context2qn_attn.build_graph(hidden[:, :self.FLAGS.question_len+1], qn_unify_mask, context_hidden) 
        ## context attention: add und attn in question and conclude und, ques_attn do not include und
        ## todo
        context_attn_output = tf.concat([tf.expand_dims(tf.add(context_attn_output[:, 0], qn_attn_output[:, -1]), 1), context_attn_output[:, 1:]], 1) # shape(batch_size, context_len, hidden_size*2)
        qn_attn_output = qn_attn_output[:, :-1] # shape(batch_size, ques_len, hiddensize*2)
        Wq = tf.get_variable(name="Wq", shape=[self.FLAGS.hidden_size*2], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        qn_weight = tf.nn.softmax(tf.einsum('aij,j->ai',qn_attn_output, Wq), axis=1) #shape (batch_size, queslen)
        c_q = tf.squeeze(tf.matmul(tf.expand_dims(tf.nn.softmax(qn_weight),1), qn_attn_output)) #shape (batch_size, self.FLAGS.hidden_size*2)
        ## cal ith word probability for start and end
        with tf.variable_scope("start_end"):
            Ws = tf.get_variable(name="Ws", shape=[self.FLAGS.hidden_size*2,self.FLAGS.hidden_size*2], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            We = tf.get_variable(name="We", shape=[self.FLAGS.hidden_size*2,self.FLAGS.hidden_size*2], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            self.answer_start_logits = tf.squeeze(tf.matmul(tf.einsum('aij,jk->aik', context_attn_output, Ws), tf.expand_dims(c_q, 2)),axis=[2])#shape (batch_size, context_len)
            # self.answer_start_logits = tf.einsum('aij,aj->ai', temp, c_q) #shape (batch_size, context_len+1)
            self.answer_end_logits = tf.squeeze(tf.matmul(tf.einsum('aij,jk->aik', context_attn_output, We), tf.expand_dims(c_q, 2)),axis=[2]) #shape (batch_size, context_len+1)
        ## todo answer verifier
            self.answer_start_prob = tf.nn.softmax(self.answer_start_logits, axis=1) #shape (batch_size, context_len)
            self.answer_end_prob = tf.nn.softmax(self.answer_end_logits, axis=1) #shape (batch_size, context_len)
        with tf.variable_scope("answer_verifier"):
            cs = tf.einsum('aij,ai->aj', context_attn_output, self.answer_start_prob) #shape (batchsize, hiddne_size*2)
            ce = tf.einsum('aij,ai->aj', context_attn_output, self.answer_end_prob) #shape (batchsize, hiddne_size*2)
            F_classifier = tf.concat([c_q, context_attn_output[:, 0], cs, ce], axis=1) #shape (batch_size, hidden_size*8)
            Wf = tf.get_variable(name="Wf", shape=[self.FLAGS.hidden_size*8], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            self.p_logit = tf.sigmoid(tf.einsum('ij,j->i', F_classifier, Wf))# shape(batch_size)
        
    def add_loss(self):
        with tf.variable_scope("loss"):
            self.loss_av = tf.reduce_mean(-(1-self.impossible)*self.p_logit - self.impossible*(1-self.p_logit))
            tf.summary.scalar('loss_av', self.loss_av)
            loss_answer_start = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.answer_start_logits, labels=self.ans_span[:, 0])
            self.loss_answer_start = tf.reduce_mean(loss_answer_start)
            tf.summary.scalar('loss_start', self.loss_answer_start)
            loss_answer_end = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.answer_end_logits, labels=self.ans_span[:, 1])
            self.loss_answer_end = tf.reduce_mean(loss_answer_end)
            tf.summary.scalar('loss_end', self.loss_answer_end)
            loss_no_answer = self.impossible*(-tf.log(self.answer_start_prob[:, 0]) - tf.log(self.answer_end_prob[:, 0]))
            self.loss_no_answer = tf.reduce_mean(loss_no_answer)
            tf.summary.scalar('loss_no_answer', self.loss_no_answer)
            self.loss = self.loss_av+self.loss_answer_start+self.loss_answer_end+self.loss_no_answer
            tf.summary.scalar('loss', self.loss)

    def run_train_iter(self, session, batch, summary_writer):
        """
        This performs a single training iteration (forward pass, loss computation, backprop, parameter update)

        Inputs:
          session: TensorFlow session
          batch: a Batch object
          summary_writer: for Tensorboard

        Returns:
          loss: The loss (averaged across the batch) for this batch.
          global_step: The current number of training iterations we've done
          param_norm: Global norm of the parameters
          gradient_norm: Global norm of the gradients
        """
        # Match up our input data with the placeholders
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        input_feed[self.impossible] = batch.impossible
        input_feed[self.keep_prob] = 1.0 - self.FLAGS.dropout # apply dropout

        # output_feed contains the things we want to fetch.
        output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]

        # Run the model
        [_, summaries, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, feed_dict = input_feed)

        # All summaries in the graph are added to Tensorboard
        summary_writer.add_summary(summaries, global_step)

        return loss, global_step, param_norm, gradient_norm


    def get_loss(self, session, batch):
        """
        Run forward-pass only; get loss.

        Inputs:
          session: TensorFlow session
          batch: a Batch object

        Returns:
          loss: The loss (averaged across the batch) for this batch
        """

        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        input_feed[self.impossible] = batch.impossible
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.loss]

        [loss] = session.run(output_feed, input_feed)

        return loss


    def get_prob_dists(self, session, batch):
        """
        Run forward-pass only; get probability distributions for start and end positions.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          probdist_start and probdist_end: both shape (batch_size, context_len)
        """
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.impossible] = batch.impossible
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.answer_start_prob, self.answer_end_logits, self.p_logit]
        [probdist_start, probdist_end, p_logit] = session.run(output_feed, input_feed)
        return probdist_start, probdist_end, p_logit


    def get_start_end_pos_poss(self, session, batch):
        """
        Run forward-pass only; get the most likely answer span.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          start_pos, end_pos: both numpy arrays shape (batch_size).
            The most likely start and end positions for each example in the batch.
        """
        # Get start_dist and end_dist, both shape (batch_size, context_len)
        start_dist, end_dist, p_logit = self.get_prob_dists(session, batch)

        # Take argmax to get start_pos and end_post, both shape (batch_size)
        start_pos = np.argmax(start_dist, axis=1)
        end_pos = np.argmax(end_dist, axis=1)
        possible = p_logit > 0.5

        return start_pos, end_pos, possible
    


    def get_dev_loss(self, session, dev_context_path, dev_qn_path, dev_ans_path, dev_impossible_path):
        """
        Get loss for entire dev set.

        Inputs:
          session: TensorFlow session
          dev_qn_path, dev_context_path, dev_ans_path: paths to the dev.{context/question/answer} data files

        Outputs:
          dev_loss: float. Average loss across the dev set.
        """
        logging.info("Calculating dev loss...")
        tic = time.time()
        loss_per_batch, batch_lengths = [], []

        # Iterate over dev set batches
        # Note: here we set discard_long=True, meaning we discard any examples
        # which are longer than our context_len or question_len.
        # We need to do this because if, for example, the true answer is cut
        # off the context, then the loss function is undefined.
        for batch in get_batch_generator(self.word2id, dev_context_path, dev_qn_path, dev_ans_path, dev_impossible_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=True):

            # Get loss for this batch
            loss = self.get_loss(session, batch)
            curr_batch_size = batch.batch_size
            loss_per_batch.append(loss * curr_batch_size)
            batch_lengths.append(curr_batch_size)

        # Calculate average loss
        total_num_examples = sum(batch_lengths)
        toc = time.time()
        print("Computed dev loss over %i examples in %.2f seconds" % (total_num_examples, toc-tic))

        # Overall loss is total loss divided by total number of examples
        dev_loss = sum(loss_per_batch) / float(total_num_examples)

        return dev_loss


    ##todo evaluate
    def check_f1_em(self, session, context_path, qn_path,ans_path, impossible_path, dataset, num_samples=100):
        #  train_f1, train_em = self.check_f1_em(session, train_context_path, train_qn_path, train_ans_path, train_impossible_path, "train", num_samples=1000)
            ##将answer改为 如果impossible 为 “”
        logging.info("Calculating F1/EM for %s examples in %s set..." % (str(num_samples) if num_samples != 0 else "all", dataset))

        f1_total = 0.
        em_total = 0.
        example_num = 0
        tic = time.time() 

        for batch in get_batch_generator(self.word2id, context_path, qn_path, ans_path, impossible_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=False):
            pred_start_pos, pred_end_pos, pred_possible = self.get_start_end_pos_poss(session, batch)

            # Convert the start and end positions to lists length batch_size
            pred_start_pos = pred_start_pos.tolist() # list length batch_size
            pred_end_pos = pred_end_pos.tolist() # list length batch_size
            pred_possible = pred_possible.tolist() # list length batch_size

            for ex_idx, (pred_ans_start, pred_ans_end, pred_poss, true_ans_tokens, true_impossible) in enumerate(zip(pred_start_pos, pred_end_pos, pred_possible, batch.ans_tokens, batch.impossible)):
                example_num += 1

                # Get the predicted answer
                # Important: batch.context_tokens contains the original words (no UNKs)
                # You need to use the original no-UNK version when measuring F1/EM
                if pred_poss:
                    if pred_ans_start == 0 or pred_ans_end == 0:
                        pred_answer = ""
                    else:
                        pred_ans_tokens = batch.context_tokens[ex_idx][pred_ans_start : pred_ans_end+1]
                        pred_answer = " ".join(pred_ans_tokens)
                else:
                    pred_answer = ""

                # Get true answer (no UNKs)
                if not true_impossible:
                    true_answer = " ".join(true_ans_tokens)
                else:
                    true_answer = ""

                # Calc F1/EM
                f1 = compute_f1(true_answer, pred_answer)
                em = compute_exact(true_answer, pred_answer)
                f1_total += f1
                em_total += em

                # Optionally pretty-print
                # if print_to_screen:
                #     print_example(self.word2id, batch.context_tokens[ex_idx], batch.qn_tokens[ex_idx], batch.ans_span[ex_idx, 0], batch.ans_span[ex_idx, 1], pred_ans_start, pred_ans_end, true_answer, pred_answer, f1, em)

                if num_samples != 0 and example_num >= num_samples:
                    break

            if num_samples != 0 and example_num >= num_samples:
                break
        
        f1_total /= example_num
        em_total /= example_num

        toc = time.time()
        logging.info("Calculating F1/EM for %i examples in %s set took %.2f seconds" % (example_num, dataset, toc-tic))

        return f1_total, em_total



    def train(self, session, train_context_path, train_qn_path, train_ans_path, train_impossible_path, dev_qn_path, dev_context_path, dev_ans_path, dev_impossible_path):
        """
        Main training loop.

        Inputs:
          session: TensorFlow session
          {train/dev}_{qn/context/ans}_path: paths to {train/dev}.{context/question/answer} data files
        """

        # Print number of model parameters
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        # We will keep track of exponentially-smoothed loss
        exp_loss = None

        # Checkpoint management.
        # We keep one latest checkpoint, and one best checkpoint (early stopping)
        checkpoint_path = os.path.join(self.FLAGS.train_dir, "qa.ckpt")
        bestmodel_dir = os.path.join(self.FLAGS.train_dir, "best_checkpoint")
        bestmodel_ckpt_path = os.path.join(bestmodel_dir, "qa_best.ckpt")
        best_dev_f1 = None
        best_dev_em = None

        # for TensorBoard
        summary_writer = tf.summary.FileWriter(self.FLAGS.train_dir, session.graph)

        epoch = 0

        logging.info("Beginning training loop...")
        while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs:
            epoch += 1
            epoch_tic = time.time()

            # Loop over batches
            for batch in get_batch_generator(self.word2id, train_context_path, train_qn_path, train_ans_path, train_impossible_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=True):

                # Run training iteration
                iter_tic = time.time()
                loss, global_step, param_norm, grad_norm = self.run_train_iter(session, batch, summary_writer)
                iter_toc = time.time()
                iter_time = iter_toc - iter_tic

                # Update exponentially-smoothed loss
                if not exp_loss: # first iter
                    exp_loss = loss
                else:
                    exp_loss = 0.99 * exp_loss + 0.01 * loss

                # Sometimes print info to screen
                if global_step % self.FLAGS.print_every == 0:
                    logging.info(
                        'epoch %d, iter %d, loss %.5f, smoothed loss %.5f, grad norm %.5f, param norm %.5f, batch time %.3f' %
                        (epoch, global_step, loss, exp_loss, grad_norm, param_norm, iter_time))

                # Sometimes save model
                if global_step % self.FLAGS.save_every == 0:
                    logging.info("Saving to %s..." % checkpoint_path)
                    self.saver.save(session, checkpoint_path, global_step=global_step)

                # Sometimes evaluate model on dev loss, train F1/EM and dev F1/EM
                if global_step % self.FLAGS.eval_every == 0:

                    # Get loss for entire dev set and log to tensorboard
                    dev_loss = self.get_dev_loss(session, dev_context_path, dev_qn_path, dev_ans_path, dev_impossible_path)
                    logging.info("Epoch %d, Iter %d, dev loss: %f" % (epoch, global_step, dev_loss))
                    write_summary(dev_loss, "dev/loss", summary_writer, global_step)


                    # Get F1/EM on train set and log to tensorboard
                    train_f1, train_em = self.check_f1_em(session, train_context_path, train_qn_path, train_ans_path, train_impossible_path, "train", num_samples=1000)
                    logging.info("Epoch %d, Iter %d, Train F1 score: %f, Train EM score: %f" % (epoch, global_step, train_f1, train_em))
                    write_summary(train_f1, "train/F1", summary_writer, global_step)
                    write_summary(train_em, "train/EM", summary_writer, global_step)


                    # Get F1/EM on dev set and log to tensorboard
                    dev_f1, dev_em = self.check_f1_em(session, dev_context_path, dev_qn_path, dev_ans_path, dev_impossible_path, "dev", num_samples=0)
                    logging.info("Epoch %d, Iter %d, Dev F1 score: %f, Dev EM score: %f" % (epoch, global_step, dev_f1, dev_em))
                    write_summary(dev_f1, "dev/F1", summary_writer, global_step)
                    write_summary(dev_em, "dev/EM", summary_writer, global_step)


                    # Early stopping based on dev EM. You could switch this to use F1 instead.
                    if best_dev_em is None or dev_em > best_dev_em:
                        best_dev_em = dev_em
                        logging.info("Saving to %s..." % bestmodel_ckpt_path)
                        self.bestmodel_saver.save(session, bestmodel_ckpt_path, global_step=global_step)


            epoch_toc = time.time()
            logging.info("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc-epoch_tic))

        sys.stdout.flush()



def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)        







