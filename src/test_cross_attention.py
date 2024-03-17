from positional_embedding_layer import PositionalEmbedding
from cross_attention import CrossAttention
import numpy as np

vocab_size = 1000
encoder_vocab_size = 1000
decoder_vocab_size = 1100
d_model = 512

encoder_embedding_layer = PositionalEmbedding(vocab_size, d_model)
decoder_embedding_layer = PositionalEmbedding(vocab_size, d_model)

random_encoder_input = np.random.randint(0, encoder_vocab_size, size=(1, 100))
random_decoder_input = np.random.randint(0, decoder_vocab_size, size=(1, 110))

encoder_embeddings = encoder_embedding_layer(random_encoder_input)
decoder_embeddings = decoder_embedding_layer(random_decoder_input)

print("encoder_embeddings shape", encoder_embeddings.shape)
print("decoder_embeddings shape", decoder_embeddings.shape)

cross_attention_layer = CrossAttention(num_heads=2, key_dim=512)
cross_attention_output = cross_attention_layer(decoder_embeddings, encoder_embeddings)

print("cross_attention_output shape", cross_attention_output.shape)