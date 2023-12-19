import numpy as np

# a=range(27)
# a=np.array(a)
# a=np.reshape(a,[3,3,3])
# print(a[2])


import tensorflow as tf

A = [1, 3, 4, 5, 6]
B = [1, 3, 4, 3, 2]

correct_prediction = tf.equal(A, B)
with tf.Session() as sess:
    # print(sess.run(tf.equal(A, B)))
    #[ True  True  True False False]
    print(sess.run(tf.equal(A, B)))
    #[1. 1. 1. 0. 0.]
    print(sess.run(tf.cast(correct_prediction, tf.float32)))
    #0.6
    print(sess.run(tf.reduce_mean(tf.cast(correct_prediction, tf.float32))))