# -*- coding: utf-8 -*-
import json
import math
from tqdm import tqdm, trange
import numpy as np
import tensorflow as tf

class PointerNetworks:
    def __init__(self, config):
        self.config = config
        self.max_length = self.config.max_length

        self.global_step = 0

        self.encoder_input_placeholder_list = []
        self.decoder_input_placeholder_list = []
        self.decoder_labels_placeholder_list = []
        self.labels_weight_list = []
        self.dropout_placeholder = None

        self.tensor_dict = {}

        self.build()

# --- model architecture ---

    def add_placeholders(self):
        """
        總共有幾個placeholders
        input_placeholder: (None, time_steps, )

        :return:
        """
        for i in range(self.max_length):
            self.encoder_input_placeholder_list += [tf.placeholder(
                dtype=tf.float32, shape=[None, self.config.encode_size], name='EncoderInput%d' % i)]

        for i in range(self.max_length + 1):
            self.decoder_input_placeholder_list += [tf.placeholder(
                dtype=tf.float32, shape=[None, self.config.decode_size], name='DecoderInput%d' % i)]
            self.decoder_labels_placeholder_list += [tf.placeholder(
                dtype=tf.int32, shape=[None], name='DecoderLabels%d' % i)]
            self.labels_weight_list += [tf.placeholder(
                dtype=tf.float32, shape=[None, 1], name='LabelsWeight%d' %i)]

        self.dropout_placeholder = tf.placeholder(dtype=tf.float32, shape=())

    def create_feed_dict(self, encoder_input_batch, decoder_input_batch, decoder_labels_batch, dropout=1):
        feed_dict = { self.dropout_placeholder:dropout }
        for placeholder, data in zip(self.encoder_input_placeholder_list, encoder_input_batch):
            feed_dict[placeholder] = data
        for placeholder, data in zip(self.decoder_input_placeholder_list, decoder_input_batch):
            feed_dict[placeholder] = data
        for placeholder, data in zip(self.decoder_labels_placeholder_list, decoder_labels_batch):
            feed_dict[placeholder] = data

        for placeholder in self.labels_weight_list:
            feed_dict[placeholder] = np.ones([len(encoder_input_batch), 1])
        return feed_dict

    def pointer_decoder(self, decoder_inputs, initial_state, attention_states, cell,
                        feed_prev=True):
        """RNN decoder with pointer net for the sequence-to-sequence model.
        Args:
          decoder_inputs: a list of 2D Tensors [batch_size x cell.input_size].
          initial_state: 2D Tensor [batch_size x cell.state_size].
          attention_states: 3D Tensor [batch_size x attn_length x attn_size].
          cell: rnn_cell.RNNCell defining the cell function and size.
          dtype: The dtype to use for the RNN initial state (default: tf.float32).
          scope: VariableScope for the created subgraph; default: "pointer_decoder".
        Returns:
          outputs: A list of the same length as decoder_inputs of 2D Tensors of shape
            [batch_size x output_size]. These represent the generated outputs.
            Output i is computed from input i (which is either i-th decoder_inputs.
            First, we run the cell
            on a combination of the input and previous attention masks:
              cell_output, new_state = cell(linear(input, prev_attn), prev_state).
            Then, we calculate new attention masks:
              new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
            and then we calculate the output:
              output = linear(cell_output, new_attn).
          states: The state of each decoder cell in each time-step. This is a list
            with length len(decoder_inputs) -- one item for each time-step.
            Each item is a 2D Tensor of shape [batch_size x cell.state_size].
        """
        if not decoder_inputs:
            raise ValueError("Must provide at least 1 input to attention decoder.")
        if not attention_states.get_shape()[1:2].is_fully_defined():
            raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                             % attention_states.get_shape())

        def linear(input_, output_size):
            shape = input_.get_shape().as_list()
            if len(shape) != 2:
                raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
            if not shape[1]:
                raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
            input_size = shape[1]

            # Now the computation.
            with tf.variable_scope("SimpleLinear"):
                matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
                bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

            return tf.matmul(input_, tf.transpose(matrix)) + bias_term

        with tf.variable_scope("point_decoder"):
            batch_size = tf.shape(decoder_inputs[0])[0]  # Needed for reshaping.
            input_size = decoder_inputs[0].get_shape()[1].value
            attn_length = attention_states.get_shape()[1].value
            attn_size = attention_states.get_shape()[2].value

            # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
            hidden = tf.reshape(
                attention_states, [-1, attn_length, 1, attn_size])

            attention_vec_size = attn_size  # Size of query vectors for attention.
            k = tf.get_variable("AttnW", [1, 1, attn_size, attention_vec_size])
            hidden_features = tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
            v = tf.get_variable("AttnV", [attention_vec_size])

            states = [initial_state]

            def attention(query):
                """Point on hidden using hidden_features and query."""
                with tf.variable_scope("attention"):
                    y = linear(query, attention_vec_size)
                    y = tf.reshape(y, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = tf.reduce_sum(
                        v * tf.nn.tanh(hidden_features + y), [2, 3])
                    return s

            outputs = []
            batch_attn_size = tf.stack([batch_size, attn_size])
            attns = tf.zeros(batch_attn_size, dtype=tf.float32)

            attns.set_shape([None, attn_size])
            inps = []
            for i in range(len(decoder_inputs)):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                inp = decoder_inputs[i]

                if feed_prev and i > 0:
                    inp = tf.stack(decoder_inputs)
                    inp = tf.transpose(inp, perm=[1, 0, 2])
                    inp = tf.reshape(inp, [-1, attn_length, input_size])
                    inp = tf.reduce_sum(inp * tf.reshape(tf.nn.softmax(output), [-1, attn_length, 1]), 1)
                    inp = tf.stop_gradient(inp)
                    inps.append(inp)

                # Use the same inputs in inference, order internaly

                # Merge input and previous attentions into one vector of the right size.
                x = linear(tf.convert_to_tensor([inp, attns], dtype=tf.float32), cell.output_size)
                # Run the RNN.
                cell_output, new_state = cell(x, states[-1])
                states.append(new_state)
                # Run the attention mechanism.
                output = attention(new_state)

                outputs.append(output)

        return outputs, states, inps

    def add_prediction_op(self):
        dropout_rate = self.dropout_placeholder

        cell = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size)
        if self.config.is_train:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_rate)

        # encoder
        with tf.variable_scope("encoder"):
            # output = list of [batch_size x hidden_size]
            N = len(self.encoder_input_placeholder_list)
            encoder_outputs, final_state = tf.nn.static_rnn(cell, self.encoder_input_placeholder_list, dtype=tf.float32)
            # add dummy element to encoder outputs
            encoder_outputs = [np.zeros(N, self.config.hidden_size)] + encoder_outputs

            states = [tf.reshape(e, shape=[-1, 1, cell.output_size]) for e in encoder_outputs]
            att_states = tf.concat(states, axis=1)
        # for updating gradients
        with tf.variable_scope("decoder"):
            outputs, states, _ = self.pointer_decoder(
                self.decoder_input_placeholder_list, final_state, att_states, cell)
        # for test loss
        with tf.variable_scope("decoder", reuse=True):
            preds, _, inps = self.pointer_decoder(
                self.decoder_input_placeholder_list, final_state, att_states, cell)

        self.preds = preds

        self.outputs = outputs
        self.inps = inps

    def add_loss_op(self, preds):
        loss = []
        for logit, label, weight in zip(preds, self.decoder_labels_placeholder_list, self.labels_weight_list):
            loss += [tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=label) * weight]
        return loss

    def add_cost_op(self, loss):
        cost = tf.reduce_mean(loss)
        return cost

    def add_training_op(self, cost):
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        # compute gradients here
        grads_and_vars = optimizer.compute_gradients(cost)
        grads, variables = zip(*grads_and_vars)
        grads, grad_norm = tf.clip_by_global_norm(grads, clip_norm=self.config.max_grad_norm)

        train_op = optimizer.apply_gradients(zip(grads, variables))
        return train_op

    def build(self):
        self.add_placeholders()
        self.add_prediction_op()
        self.loss = self.add_loss_op(self.outputs)
        self.test_loss = self.add_loss_op(self.preds)
        self.cost = self.add_cost_op(self.loss)
        self.test_cost = self.add_cost_op(self.test_loss)
        self.inps_mean = self.add_cost_op(self.inps)
        if self.config.is_train:
            self.train_op = self.add_training_op(self.cost)
            self.sum_op = self.add_summary_op()

    def add_summary_op(self):
        tf.summary.scalar('cost', self.cost)
        tf.summary.scalar('test_cost', self.test_cost)
        tf.summary.scalar('inps', self.inps_mean)
        merged = tf.summary.merge_all()
        return merged

# --- model operation ---

    def train_on_batch(self, sess, encoder_input_batch, decoder_input_batch, decoder_labels_batch):
        feed = self.create_feed_dict(encoder_input_batch, decoder_input_batch, decoder_labels_batch)
        _, cost, _, _, summary = sess.run([self.train_op, self.cost, self.test_cost, self.inps_mean, self.sum_op], feed_dict=feed)
        return cost, summary

    def run_epoch(self, sess, data_loader, writer):
        total_loss = []
        x1_train, x2_train, y2_train = data_loader.get_train_data()
        n_batch = int(math.ceil(float(len(x1_train)) / self.config.batch_size))
        pbar = trange(n_batch)
        for i in pbar:
            batch = (x1_train[i * self.config.batch_size: (i+1) * self.config.batch_size],
                     x2_train[i * self.config.batch_size: (i+1) * self.config.batch_size],
                     y2_train[i * self.config.batch_size: (i+1) * self.config.batch_size])

            cost, summary = self.train_on_batch(sess, *batch)
            total_loss += [cost]
            writer.add_summary(summary, global_step=self.global_step)
            pbar.set_description("train loss = {}".format(cost))
            self.global_step += 1

        return np.mean(total_loss)

    def fit(self, sess, saver, data_loader):
        writer = tf.summary.FileWriter(self.config.log_output + '/train',
                                             sess.graph)
        for epoch in trange(self.config.n_epochs):
            loss = self.run_epoch(sess, data_loader, writer)
            tf.logging.info("Epoch %d out of %d, loss = %.4f", epoch + 1, self.config.n_epochs, loss)
            if saver:
                saver.save(sess, self.config.model_output)