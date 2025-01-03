import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
a = tf.constant([1.0, 2.0, 3.0])
b = tf.constant([4.0, 5.0, 6.0])
c = a + b
print("Result", c)