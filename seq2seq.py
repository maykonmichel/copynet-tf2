import tensorflow as tf

from tensorflow.keras import Model, Sequential
from copynet import Encoder, CopyDecoder

class CopySeq2Seq(Model):
    def __init__(self, vocab_size, embed_size, hidden_size, oov_idx, max_oovs=12):
        super(CopySeq2Seq, self).__init__()
        self.encoder = Encoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            oov_idx=oov_idx)
        self.decoder = CopyDecoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            oov_idx=oov_idx,
            max_oovs=max_oovs)
    def call(self, x_enc, x_dec, training=False):
        enc_states = self.encoder(x_enc)
        if training:
            probs, states = self.decoder(x_dec, enc_states, x_enc, training=True)
            return probs, states
        else:
            probs, states, preds = self.decoder(x_dec, enc_states, x_enc, training=False)
            return probs, states, preds