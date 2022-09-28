
import tensorflow as tf
import numpy as np

print(tf.__version__)

#계산 그래프 연산
x_vals = np.array([1., 3., 5., 7., 9.])

m_const = tf.constant(3.)

for x_val in x_vals:
    tf.print(tf.multiply(m_const, x_val))

#다중 연산 중첩
my_array = np.array([[1., 3., 5., 7., 9.], [-2., 0., 2., 4., 6.], [-6., -3., 0., 3., 6.]], dtype='float64')
x_vals = np.array([my_array, my_array +1])
print(x_vals.dtype)
m1 = tf.constant([[1.], [0.], [-1.], [2.], [4.]], dtype='float64')
m2 = tf.constant([[2.]], dtype='float64')
a1 = tf.constant([[10.]], dtype='float64')

print(m1.dtype)

def multiple_func(a):
    prod1 = tf.matmul(a, m1)
    prod2 = tf.matmul(prod1, m2)
    add1 = tf.add(prod2, a1)
    return add1

for x_val in x_vals:
    tf.print(multiple_func(x_val))

#다층 처리
x_shape = [1, 4, 4, 1]
x_val = np.random.uniform(size=x_shape)
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

tf.print(custom_layer1(x_val))