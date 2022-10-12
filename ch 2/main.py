
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging


# print(tf.__version__)
#
# #계산 그래프 연산
# x_vals = np.array([1., 3., 5., 7., 9.])
#
# m_const = tf.constant(3.)
#
# for x_val in x_vals:
#     tf.print(tf.multiply(m_const, x_val))
#
# #다중 연산 중첩
# my_array = np.array([[1., 3., 5., 7., 9.], [-2., 0., 2., 4., 6.], [-6., -3., 0., 3., 6.]], dtype='float64')
# x_vals = np.array([my_array, my_array +1])
# print(x_vals.dtype)
# m1 = tf.constant([[1.], [0.], [-1.], [2.], [4.]], dtype='float64')
# m2 = tf.constant([[2.]], dtype='float64')
# a1 = tf.constant([[10.]], dtype='float64')
#
# print(m1.dtype)

def multiple_func(a):
    prod1 = tf.matmul(a, m1)
    prod2 = tf.matmul(prod1, m2)
    add1 = tf.add(prod2, a1)
    return add1

def huber(y_true, y_pred, delta):
    error = y_true - y_pred
    se = tf.abs(error) < delta
    sq_loss = tf.square(error) / 2
    linear_loss = delta * tf.abs(error) - (delta**2)/2
    return tf.where(se, sq_loss, linear_loss)

def mov_layer(x_data):
    my_filter = tf.constant(0.25, shape=[2, 2, 1, 1])
    my_strides = [1, 2, 2, 1]
    mov_avg_layer = tf.nn.conv2d(x_data, my_filter, my_strides, padding='SAME', name='Moving_Avg_Window')
    return mov_avg_layer

def custom_layer(input_matrix):
    input_matrix_sqeezed = tf.squeeze(input_matrix)
    A = tf.constant([[1., 2.], [-1., 3.]])
    b = tf.constant(1., shape=[2, 2])
    temp1 = tf.matmul(A, input_matrix_sqeezed)
    temp = tf.add(temp1, b) # Ax+b
    return (tf.sigmoid(temp))

with tf.name_scope('Custom_Layer') as scope:
    def custom_layer1(x_data):
        return custom_layer(mov_layer(x_data))

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fah_q = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

for i, c in enumerate(celsius_q):
    print("{} degrees Celsius = {} degrees Fahrenghet".format(c, fah_q[i]))

l0 = tf.keras.layers.Dense(units= 1, input_shape=[1])

model = tf.keras.Sequential([l0]) #ML을 위한 모델 형성

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1)) #0,1 은 learning rate, model 컴파일, 로스 측정 및 옵티마이징 방법 명명

history = model.fit(celsius_q, fah_q, epochs=500, verbose=False)

plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(history.history['loss'])
plt.show()

print(model.predict([100.0]))