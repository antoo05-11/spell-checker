from positional_embedding_layer import PositionalEmbedding
import numpy as np
from feed_forward_layer import FeedForward

encoder_vocab_size = 1000
vocab_size = 1000
d_model = 512

encoder_embedding_layer = PositionalEmbedding(vocab_size, d_model)

random_encoder_input = np.random.randint(0, encoder_vocab_size, size=(1, 100))

encoder_embeddings = encoder_embedding_layer(random_encoder_input)

print("encoder_embeddings shape", encoder_embeddings.shape)

feed_forward_layer = FeedForward(d_model, dff=2048)
feed_forward_output = feed_forward_layer(encoder_embeddings)

print("feed_forward_output shape", feed_forward_output.shape)
