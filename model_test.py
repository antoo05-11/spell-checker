import tensorflow as tf
import tensorflow_text

reloaded = tf.saved_model.load('translator')

print((str(reloaded('este').numpy()).split(' ')))