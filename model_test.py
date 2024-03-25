import tensorflow as tf

reloaded = tf.saved_model.load('translator')

print(reloaded('este é o primeiro livro que eu fiz.').numpy())