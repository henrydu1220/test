import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
start = time.time()
x_data = np.random.rand(10000).astype(np.float32)
y_data = x_data * -82813.5 + 70146.4
W = tf.Variable(tf.random_uniform([1], -100000.0, 100000.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.2)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for step in range(20001):
    sess.run(train)
    if step % 8000 == 0:
        print(step, sess.run(W), sess.run(b))
        plt.plot(x_data, y_data, 'ro', label='Original data')
        plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label='Fitted line')
        plt.legend()
        plt.show()
end = time.time()
print end-start
