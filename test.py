import numpy as np
import tensorflow as tf

from seq2seq import CopySeq2Seq

############################
# Importante: 
# - usar PAD com indice 0
# - max_oovs corresponde ao número máximo de oovs que o modelo vai considerar
# copiar ou não, para cada sentença. Ou seja, o máximo de 
# elementos oov na sequência de entrada, que poderão ser copiados na saída.
# - oov_idx corresponde ao index para o embedding de oov. É importante notar
# que tokens com index >= vocab_size receberão embedding oov automaticamente.
# Para que possam ser considerados pelo mecanismo de cópia, seu index deve estar entre vocab_size
# e vocab_size + max_oovs. Note que, neste caso específico, o index não precisa estar atrelado 
# a um único token para todo o dataset, e sim, apenas no domínio de um par entrada/saída. 
# É um "fake" index para que a rede saiba sua posição temporal no encoder e para possivelmente 
# mapea-lo na saída.
# - quando model for chamado com parametro training=True, o retorno será (probs, states), do contrário,
# vai prever dinamicamente a saída e o retorno será (probs, states, preds). Neste ultimo caso, decoder_inputs
# deve conter apenas o token sos, com shape [batch_size x 1]
############################

vocab_size = 6
embed_size = 16
hidden_size = 32
seq_len = 4
batch_size = 3
max_oovs = 2

model = CopySeq2Seq(vocab_size, embed_size, hidden_size, oov_idx=5, max_oovs=max_oovs)

batch_x_enc = np.array([[2,3,4,5], [2,3,0,0], [4,5,6,0]]) # aqui, por ex, 6 está fora do vocab_size
batch_x_dec = np.array([[1,2,3,4], [1,2,3,0], [1,4,5,6]])
batch_y = np.array([[2,3,4,5], [2,3,0,0], [4,5,6,0]])

optimizer = tf.keras.optimizers.Adam(0.001)
cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

@tf.function
def train_step(x_enc, x_dec, y):
    with tf.GradientTape() as tape:
        probs, _ = model(x_enc, x_dec, training=True)
        target = tf.one_hot(y, depth=vocab_size+max_oovs, axis=-1, dtype=tf.float32)
        losses = cce(target, probs)
        loss= (1/batch_size) * tf.reduce_sum(losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
    return probs, target, loss

@tf.function
def infer_step(x_enc, sos_idx):
    sos = np.array([[sos_idx], [sos_idx], [sos_idx]])
    _, _, preds = model(x_enc, sos, training=False)
  
    return preds

for i in range(500):
    probs, target, loss= train_step(batch_x_enc, batch_x_dec, batch_y)
    # print(loss.numpy())

print(infer_step(batch_x_enc, 1).numpy())
print(batch_y)