from positional_embedding_layer import PositionalEmbedding
from casual_self_attention import CausalSelfAttention
import numpy as np
import tensorflow as tf

vocab_size = 1000
decoder_vocab_size = 1100
d_model = 512

decoder_embedding_layer = PositionalEmbedding(vocab_size, d_model)

random_decoder_input = np.random.randint(0, decoder_vocab_size, size=(1, 110))

decoder_embeddings = decoder_embedding_layer(random_decoder_input)

print("decoder_embeddings shape", decoder_embeddings.shape)

causal_self_attention_layer = CausalSelfAttention(num_heads=2, key_dim=512)
causal_self_attention_output = causal_self_attention_layer(decoder_embeddings)

print("causal_self_attention_output shape", causal_self_attention_output.shape)

out1 = causal_self_attention_layer(decoder_embedding_layer(random_decoder_input[:, :50])) # Only the first 50 tokens beffore applying the embedding layer
out2 = causal_self_attention_layer(decoder_embedding_layer(random_decoder_input)[:, :50]) # Only the first 50 tokens after applying the embedding layer

diff = tf.reduce_max(tf.abs(out1 - out2)).numpy()

print("Difference between the two outputs:", diff)