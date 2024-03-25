import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text

from PositionalEmbedding import PositionalEmbedding
from CrossAttention import CrossAttention
from GlobalSelfAttention import GlobalSelfAttention
from CasualSelfAttention import CausalSelfAttention
from FeedForward import FeedForward
from EncoderLayer import EncoderLayer
from Encoder import Encoder
from DecoderLayer import DecoderLayer
from Decoder import Decoder
from Transformer import Transformer
from ExportTranslator import ExportTranslator
from Translator import Translator
from masked_utils import masked_accuracy, masked_loss
from CustomSchedule import CustomSchedule

# Download dataset with pt - en pairs.
examples, metadata = tfds.load(
    "ted_hrlr_translate/pt_to_en", with_info=True, as_supervised=True
)

train_examples, val_examples = examples["train"].take(100), examples[
    "validation"
].take(100)

# Get tokenizer for both languages.
model_name = "ted_hrlr_translate_pt_en_converter"
tf.keras.utils.get_file(
    f"{model_name}.zip",
    f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
    cache_dir=".",
    cache_subdir="",
    extract=True,
)
tokenizers = tf.saved_model.load(model_name)

#
MAX_TOKENS = 128


def prepare_batch(pt, en):
    pt = tokenizers.pt.tokenize(pt)  # Output is ragged.
    pt = pt[:, :MAX_TOKENS]  # Trim to MAX_TOKENS.
    pt = pt.to_tensor()  # Convert to 0-padded dense Tensor

    en = tokenizers.en.tokenize(en)
    en = en[:, : (MAX_TOKENS + 1)]
    en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
    en_labels = en[:, 1:].to_tensor()  # Drop the [START] tokens

    return (pt, en_inputs), en_labels


#
BUFFER_SIZE = 20000
BATCH_SIZE = 64


def make_batches(ds):
    return (
        ds.shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(prepare_batch, tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


# Create training and validation set batches.
train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

for (pt, en), en_labels in train_batches.take(1):
    break

embed_pt = PositionalEmbedding(vocab_size=tokenizers.pt.get_vocab_size(), d_model=512)
embed_en = PositionalEmbedding(vocab_size=tokenizers.en.get_vocab_size(), d_model=512)

pt_emb = embed_pt(pt)
en_emb = embed_en(en)

sample_ca = CrossAttention(num_heads=2, key_dim=512)
sample_gsa = GlobalSelfAttention(num_heads=2, key_dim=512)
sample_csa = CausalSelfAttention(num_heads=2, key_dim=512)

sample_ffn = FeedForward(512, 2048)
sample_encoder_layer = EncoderLayer(d_model=512, num_heads=8, dff=2048)

sample_encoder = Encoder(
    num_layers=4, d_model=512, num_heads=8, dff=2048, vocab_size=8500
)

sample_encoder_output = sample_encoder(pt, training=False)

sample_decoder_layer = DecoderLayer(d_model=512, num_heads=8, dff=2048)

sample_decoder_layer_output = sample_decoder_layer(x=en_emb, context=pt_emb)
sample_decoder = Decoder(
    num_layers=4, d_model=512, num_heads=8, dff=2048, vocab_size=8000
)

output = sample_decoder(x=en, context=pt_emb)

# Configure transformer.
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
    dropout_rate=dropout_rate,
)
output = transformer((pt, en))

# Compile transformer model.
learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
)

transformer.compile(loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy])

transformer.fit(train_batches, epochs=50, validation_data=val_batches)

translator = Translator(tokenizers, transformer)

translator = ExportTranslator(translator)

tf.saved_model.save(translator, export_dir="translator")
