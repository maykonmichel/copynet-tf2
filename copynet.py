import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.activations import tanh
from tensorflow.keras.layers import Dense, Embedding, GRU

class Encoder(Model):
    def __init__(self, vocab_size, embed_size, hidden_size, oov_idx):
        super(Encoder, self).__init__()
        self.embed = Embedding(vocab_size, embed_size)
        self.gru = tf.keras.layers.Bidirectional(GRU(hidden_size, return_sequences=True))
        self.vocab_placeholder = tf.constant([vocab_size], tf.int32)
        self.oov_idx = oov_idx
        assert oov_idx < vocab_size, 'oov_idx must be less than vocab_size'

    def call(self, x):
        x = tf.cast(x, tf.int32) # assert x and vocab_placeholder have the same type
        # set as oov when predictions are unknown encoder inputs
        x = tf.where(
            tf.less(x, self.vocab_placeholder),
            x,
            self.oov_idx)
        embedded = self.embed(x)
        out = self.gru(embedded) # out: [batch_size x seq x hidden*2] (biRNN)
        return out 

class TimestepDecoder(Model):
    def __init__(self, vocab_size, embed_size, hidden_size, max_oovs=12):
        super(TimestepDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed = Embedding(vocab_size, embed_size)
        self.gru = GRU(hidden_size, return_sequences=True)
        self.max_oovs = max_oovs # largest number of OOVs available per sample

        self.Ws = Dense(hidden_size) # only used at initial stage
        self.Wo = Dense(vocab_size) # generate mode
        self.Wc = Dense(hidden_size) # copy mode

    def call(self, inputs, encoder_states, encoder_inputs, init_step=False, prev_state=None, weighted=None):
        # inputs(y_(t-1)): [batch_size]			<- idx of next input to the decoder
        # encoder_states: [batch_size x seq x hidden*2]		<- hidden states created at encoder
        # encoder_inputs: [batch_size x seq]			<- idx of inputs used at encoder
        # prev_state(s_(t-1)): [1 x batch_size x hidden]		<- hidden states to be used at decoder
        # weighted: [batch_size x 1 x hidden*2]		<- weighted attention of previous state, init with all zeros

        batch_size = encoder_states.shape[0]
        seq_len = encoder_states.shape[1]
        vocab_size = self.vocab_size
        hidden_size = self.hidden_size
        inputs = tf.cast(inputs, tf.int32)
        encoder_inputs = tf.cast(encoder_inputs, tf.int32)

        assert batch_size == inputs.shape[0], 'encoder and decoder inputs must have same sequence batch size'

        if init_step==True:
            prev_state = self.Ws(encoder_states[:,-1]) # picks last state
            # prev_state = tf.expand_dims(prev_state, 0)
            weighted = tf.zeros((batch_size, 1, hidden_size*2))

        gru_input = tf.concat([tf.expand_dims(self.embed(inputs), 1), weighted], 2)
        state = self.gru(gru_input, initial_state=prev_state)
        state = tf.squeeze(state)

        score_g = self.Wo(state)

        score_c = tanh(self.Wc(tf.reshape(encoder_states, (-1, hidden_size*2)))) # [batch_size*seq x hidden_size]
        score_c = tf.reshape(score_c, (batch_size,-1,hidden_size)) # [batch_size x seq x hidden_size]
        score_c = tf.squeeze(tf.matmul(score_c, tf.expand_dims(state, 2))) # batch multpication -> [batch_size x seq]
        score_c = tanh(score_c)

        encoder_mask = tf.cast(encoder_inputs == 0, tf.float32)*(-1000) # in order to work, PAD idx MUST be 0
        score_c = score_c + encoder_mask # [batch_size x seq]; padded parts will get close to 0 when applying softmax

        score = tf.concat([score_g, score_c],1) # [batch_size x (vocab + seq)]
        probs = tf.nn.softmax(score)
        prob_g = probs[:,:vocab_size] # [batch_size x vocab]
        prob_c = probs[:,vocab_size:] # [batch_size x seq]

        if self.max_oovs > 0:
            oovs = tf.zeros((batch_size, self.max_oovs), tf.float32) + 1e-5
            prob_g = tf.concat([prob_g, oovs], 1)

        one_hot = tf.one_hot(encoder_inputs, depth=vocab_size+self.max_oovs, axis=-1) # one hot tensor: [batch_size x seq x vocab+max_oovs]
        prob_c_to_g = tf.matmul(tf.expand_dims(prob_c, 1), one_hot) # [batch_size x 1 x vocab+max_oovs]
        prob_c_to_g = tf.squeeze(prob_c_to_g) # [batch_size x vocab+max_oovs]

        out = prob_g + prob_c_to_g # [batch_size x vocab+max_oovs]
        # out = tf.expand_dims(out, 1) # [batch_size x 1 x vocab+max_oovs]

        repeat_inputs = tf.tile(tf.expand_dims(inputs, 1), multiples=(1, seq_len))
        idx_from_input = tf.equal(encoder_inputs, repeat_inputs)
        idx_from_input = tf.reduce_any(idx_from_input, axis=1) # shows whether each decoder input has previously appeared in the encoder
        idx_from_input = tf.cast(idx_from_input, tf.float32)
        idx_from_input = tf.expand_dims(idx_from_input, 1)

        attn = prob_c * idx_from_input
        attn = tf.expand_dims(attn, 1) # [batch_size x 1 x seq]
        weighted = tf.matmul(attn, encoder_states)

        return out, state, weighted

class CopyDecoder(Model):
    def __init__(self, vocab_size, embed_size, hidden_size, oov_idx, max_oovs=12):
        super(CopyDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.oov_idx = oov_idx
        assert oov_idx < vocab_size, 'oov_idx must be less than vocab_size'

        self.vocab_placeholder = tf.constant([self.vocab_size], tf.int32)

        self.decoder_step = TimestepDecoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            max_oovs=max_oovs)

    @tf.function
    def call(self, decoder_inputs, encoder_states, encoder_inputs, training=False):
        # decoder_inputs: [batch_sie, max_len] <- idx of decoder inputs
        # encoder_states: [batch_size x seq x hidden*2] <- hidden states created at encoder
        # encoder_inputs: [batch_size x seq]			<- idx of inputs used at encoder
        
        max_seq_len = encoder_inputs.shape[1]

        decoder_inputs = tf.cast(decoder_inputs, tf.int32) # assert x and vocab_placeholder have the same type
        # set as oov when predictions are unknown encoder inputs
        decoder_inputs = tf.where(
            tf.less(decoder_inputs, self.vocab_placeholder),
            decoder_inputs,
            self.oov_idx)

        probs = tf.TensorArray(tf.float32, size=max_seq_len)
        states = tf.TensorArray(tf.float32, size=max_seq_len)

        # first decoding step
        prob, state, weighted = self.decoder_step(
            inputs=decoder_inputs[:, 0],
            encoder_states=encoder_states,
            encoder_inputs=encoder_inputs,
            init_step=True)  
        states = states.write(0, state)
        probs = probs.write(0, prob)
        
        if training: # second step and forward
            for i in tf.range(1, max_seq_len):
                prob, state, weighted = self.decoder_step(
                    inputs=decoder_inputs[:, i],
                    encoder_states=encoder_states,
                    encoder_inputs=encoder_inputs,
                    init_step=False,
                    prev_state=state,
                    weighted=weighted)
                states = states.write(i, state)
                probs = probs.write(i, prob)
            probs = tf.transpose(probs.stack(), [1, 0, 2])
            states = tf.transpose(states.stack(), [1, 0, 2])
            return probs, states
        else:
            preds = tf.TensorArray(tf.int32, size=max_seq_len)
            pred = tf.argmax(prob, axis=-1, output_type=tf.int32)
            preds = preds.write(0, pred)
            # set as oov when predictions are unknown copies of encoder inputs
            dec_input = tf.where(
                tf.less(pred, self.vocab_placeholder),
                pred,
                self.oov_idx)
            for i in tf.range(1, max_seq_len):
                prob, state, weighted = self.decoder_step(
                    inputs=dec_input,
                    encoder_states=encoder_states,
                    encoder_inputs=encoder_inputs,
                    init_step=False,
                    prev_state=state,
                    weighted=weighted)
                states = states.write(i, state)
                probs = probs.write(i, prob)
                pred = tf.argmax(prob, axis=-1, output_type=tf.int32)
                preds = preds.write(i, pred)
                # set as oov when predictions are unknown copies of encoder inputs
                dec_input = tf.where(
                    tf.less(pred, self.vocab_placeholder),
                    pred,
                    self.oov_idx)
            probs = tf.transpose(probs.stack(), [1, 0, 2])
            states = tf.transpose(states.stack(), [1, 0, 2])
            preds = tf.transpose(preds.stack(), [1, 0])
            return probs, states, preds