# This is a sample tensorflow coding for devduck

import sys
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import numpy as np

def custom_polynomial(value):
    return (tf.subtract(3 * tf.square(value), value) + 10)

#텐서 초기화
zero_tsr = tf.zeros([5,5])#0 값 채우는 텐서
ones_tsr = tf.ones([5,5])#1로 채우는 텐서
#순열 텐서
linear_tsr = tf.linspace(start=0, stop=1, num=3)#순열 텐서
linear_seq_tsr = tf.range(start=6, limit=15, delta=3)#2번째 정의방법(마지막 경계 값은 포함되지 않음)
#랜덤 텐서
randunif_tsr = tf.random.uniform([2,2], minval=0, maxval =1)#균등 분포 난수
randnorm_tsr = tf.random.normal([2,2], mean=0.0, stddev = 1.0)#정규 분포 난수
runcnorm_tsr = tf.random.truncated_normal([2,2], mean=0.0, stddev = 1.0)# 표준편차 2배 이내의 정규 분포 난수

my_var = tf.Variable(tf.zeros([2,2]))
print(my_var)

#플레이스 홀더 : 특정 "타입"과 "형태"의 데이터를 투입하게 될 객체 -> tensorflow 2.0 에서 제거됨(session 정의도 이제 필요 없음)
my_var = tf.Variable(tf.zeros([2, 3]))
sess=tf1.Session()
tf.compat.v1.disable_eager_execution()
x = tf1.placeholder(tf1.float32, shape=[2, 2])
y = tf.identity(x)
x_vals = np.random.rand(2, 2)
sess.run(y, feed_dict={x: x_vals})

identity_matrix = tf.linalg.diag([1.0, 1.0, 1.0])
A = tf.random.truncated_normal([2, 3])
B = tf.fill([2, 3], 5.0)
C = tf.random.uniform([3,2])
D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
print(sess.run(identity_matrix))
print(sess.run(A))

print(sess.run(tf.truediv(3,4)))
print(sess.run(tf.math.floordiv(3.0, 4.0)))
print(sess.run(tf.math.mod(22.0, 5.0)))
print(sess.run(tf.linalg.cross([1., 0., 0.], [0., 1., 0.])))

print(sess.run(custom_polynomial(11)))
sess.close()

print(tf.divide(3, 4))
print(tf.multiply(3, 4))
tf.print(tf.truediv(3, 4), output_stream=sys.stderr)
