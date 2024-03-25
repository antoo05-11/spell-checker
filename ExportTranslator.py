import tensorflow as tf


class ExportTranslator(tf.Module):
    def __init__(self, translator):
        self.translator = translator

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        MAX_TOKENS = 128

        (result, tokens, attention_weights) = self.translator(
            sentence, max_length=MAX_TOKENS
        )

        return result
