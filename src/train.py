from model import Transformer
from configs import ModelConfigs
from custom_tokenizer import CustomTokenizer

import numpy as np

import tensorflow as tf

try:
    [
        tf.config.experimental.set_memory_growth(gpu, True)
        for gpu in tf.config.experimental.list_physical_devices("GPU")
    ]
except:
    pass

from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from mltu.tensorflow.callbacks import Model2onnx, WarmupCosineDecay

from mltu.tensorflow.dataProvider import DataProvider
from mltu.tokenizers import CustomTokenizer

from mltu.tensorflow.transformer.utils import MaskedAccuracy, MaskedLoss
from mltu.tensorflow.transformer.callbacks import EncDecSplitCallback

configs = ModelConfigs()

# Path to dataset
en_training_data_path = "../data/en-es/opus.en-es-train.en"
en_validation_data_path = "../data/en-es/opus.en-es-dev.en"
es_training_data_path = "../data/en-es/opus.en-es-train.es"
es_validation_data_path = "../data/en-es/opus.en-es-dev.es"


def read_files(path):
    with open(path, "r", encoding="utf-8") as f:
        en_train_dataset = f.read().split("\n")[:-1]
    return en_train_dataset


en_training_data = read_files(en_training_data_path)
en_validation_data = read_files(en_validation_data_path)
es_training_data = read_files(es_training_data_path)
es_validation_data = read_files(es_validation_data_path)

# Consider only sentences with length <= 500
max_lenght = 500
train_dataset = [
    [es_sentence, en_sentence]
    for es_sentence, en_sentence in zip(es_training_data, en_training_data)
    if len(es_sentence) <= max_lenght and len(en_sentence) <= max_lenght
]
val_dataset = [
    [es_sentence, en_sentence]
    for es_sentence, en_sentence in zip(es_validation_data, en_validation_data)
    if len(es_sentence) <= max_lenght and len(en_sentence) <= max_lenght
]
es_training_data, en_training_data = zip(*train_dataset)
es_validation_data, en_validation_data = zip(*val_dataset)

# prepare spanish tokenizer, this is the input language
tokenizer = CustomTokenizer(char_level=True)
tokenizer.fit_on_texts(es_training_data)
tokenizer.save(configs.model_path + "/tokenizer.json")

# prepare english tokenizer, this is the output language
detokenizer = CustomTokenizer(char_level=True)
detokenizer.fit_on_texts(en_training_data)
detokenizer.save(configs.model_path + "/detokenizer.json")


def preprocess_inputs(data_batch, label_batch):
    encoder_input = np.zeros((len(data_batch), tokenizer.max_length)).astype(np.int64)
    decoder_input = np.zeros((len(label_batch), detokenizer.max_length)).astype(
        np.int64
    )
    decoder_output = np.zeros((len(label_batch), detokenizer.max_length)).astype(
        np.int64
    )

    data_batch_tokens = tokenizer.texts_to_sequences(data_batch)
    label_batch_tokens = detokenizer.texts_to_sequences(label_batch)

    for index, (data, label) in enumerate(zip(data_batch_tokens, label_batch_tokens)):
        encoder_input[index][: len(data)] = data
        decoder_input[index][: len(label) - 1] = label[:-1]  # Drop the [END] tokens
        decoder_output[index][: len(label) - 1] = label[1:]  # Drop the [START] tokens

    return (encoder_input, decoder_input), decoder_output


# Create Training Data Provider
train_dataProvider = DataProvider(
    train_dataset,
    batch_size=configs.batch_size,
    batch_postprocessors=[preprocess_inputs],
    use_cache=True,
)

# Create Validation Data Provider
val_dataProvider = DataProvider(
    val_dataset,
    batch_size=configs.batch_size,
    batch_postprocessors=[preprocess_inputs],
    use_cache=True,
)

# Create TensorFlow Transformer Model
transformer = Transformer(
    num_layers=configs.num_layers,
    d_model=configs.d_model,
    num_heads=configs.num_heads,
    dff=configs.dff,
    input_vocab_size=len(tokenizer) + 1,
    target_vocab_size=len(detokenizer) + 1,
    dropout_rate=configs.dropout_rate,
    encoder_input_size=tokenizer.max_length,
    decoder_input_size=detokenizer.max_length,
)

transformer.summary()
