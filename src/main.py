import mltu.tensorflow.dataProvider
import numpy as np
import os
from custom_tokenizer import CustomTokenizer
from configs import ModelConfigs
from mltu.tensorflow.dataProvider import DataProvider
from transformer import Transformer

file_dir = os.path.join(os.getcwd(), "data")


def read_data(path):
    sentences = []
    file = open(path, "r")
    for line in file:
        sentences.append(line.strip())
    return sentences


en_train_data = read_data(path=os.path.join(file_dir, "en-es/opus.en-es-train.en"))
es_train_data = read_data(path=os.path.join(file_dir, "en-es/opus.en-es-train.es"))
en_val_data = read_data(path=os.path.join(file_dir, "en-es/opus.en-es-dev.en"))
es_val_data = read_data(path=os.path.join(file_dir, "en-es/opus.en-es-dev.es"))


tokenizer = CustomTokenizer(char_level=True)
detokenizer = CustomTokenizer(char_level=True)

tokenizer.fit_on_texts(es_train_data)
tokenizer.save("model/tokenize/tokenizer.json")

detokenizer.fit_on_texts(en_train_data)
detokenizer.save("model/tokenize/detokenizer.json")

# sentences = tokenizer.texts_to_sequences(['Hello world, how are you?'])
# print(sentences)

configs = ModelConfigs()

max_length = 500
train_dataset = [
    [es_sentence, en_sentence]
    for es_sentence, en_sentence in zip(es_train_data, en_train_data)
    if len(es_sentence) <= max_length and len(en_sentence) <= max_length
]
val_dataset = [
    [es_sentence, en_sentence]
    for es_sentence, en_sentence in zip(es_val_data, en_val_data)
    if len(es_sentence) <= max_length and len(en_sentence) <= max_length
]
es_training_data, en_training_data = zip(*train_dataset)
es_validation_data, en_validation_data = zip(*val_dataset)


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


# print(preprocess_inputs(es_train_data[0], en_train_data[0])[0][1])

# print(preprocess_inputs(es_train_data[0], en_train_data[0])[0][0])

# print(preprocess_inputs(es_train_data[0], en_train_data[0])[1])

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
    input_vocab_size=len(tokenizer)+1,
    target_vocab_size=len(detokenizer)+1,
    dropout_rate=configs.dropout_rate,
    encoder_input_size=tokenizer.max_length,
    decoder_input_size=detokenizer.max_length
    )

transformer.summary()