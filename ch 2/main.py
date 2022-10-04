
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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

def huber(y_true, y_pred, delta):
    error = y_true - y_pred
    se = tf.abs(error) < delta
    sq_loss = tf.square(error) / 2
    linear_loss = delta * tf.abs(error) - (delta**2)/2
    return tf.where(se, sq_loss, linear_loss)


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

x_vals = tf.linspace(-1., 1., 500)
target = tf.constant(0.)

#L2 norm 비용함수(유클리드 비용 함수)
l2_y_vals = tf.square(target - x_vals)
l1_y_vals = tf.abs(target - x_vals)

#pseudo-Huber 함수
delta1 = tf.constant(0.25)
phuber1_y_vals = tf.multiply(tf.square(delta1), tf.sqrt(1. + tf.square((target - x_vals)/delta1)) -1.)
huber1_y_vals = huber(target, x_vals, delta1)
delta2 = tf.constant(5.)
phuber2_y_vals = tf.multiply(tf.square(delta2), tf.sqrt(1. + tf.square((target - x_vals)/delta2)) -1.)
huber2_y_vals = huber(target, x_vals, delta2)



x_vals2 = tf.linspace(-3., 5., 500)
target = tf.constant(1.)
targets = tf.fill([500,], 1.)

#힌지 비용 함수
hinge_y_vals = tf.maximum(0., 1. - tf.multiply(target, x_vals2))

#교차 엔트로피(cross-entropy) 비용 함수
xentropy_y_vals = -tf.multiply(target, tf.math.log(x_vals2)) - tf.multiply((1. - target), tf.math.log(1. - x_vals2))

#시그모이드 교차 엔트로피 (sigmoid cross entropy) - 교차 엔트로피에 넣기 전에 시그모이드 함수로 변환
x_val_input = tf.expand_dims(x_vals2, 1)
target_input = tf.expand_dims(targets, 1)
xentropy_sigmoid_y_vals = tf.nn.sigmoid_cross_entropy_with_logits(labels = target_input, logits = x_val_input)

#가중 교차 엔트로피 (weighted cross entropy) - sigmoid cross entropy에 가중치를 더한 것
#양수 대상 값에 가중치를 부여하며, 아래는 양수 대상 값에 0.5의 가중치를 더한것
weight = tf.constant(0.5)
xentropy_weighted_y_vals = tf.nn.weighted_cross_entropy_with_logits(targets, x_vals2, weight)

#소프트 맨스 교차 엔트로피(softmax cross entropy) -여럿이 아닌 하나의 분류 대상에 대한 비용 측정 시 사용, 확률 분포로 변환하고 실제 확률 분포와 비교 및 비용 계산
unscaled_logits = tf.constant([[1., -3., 10.]])
target_dist = tf.constant([[0.1, 0.02, 0.88]])
softmax_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits = unscaled_logits, labels= target_dist)
tf.print(softmax_xentropy)

#희소 소프트맥스 교차 엔트로피 비용 함수 (sparse softmax cross entropy) - 확률 분포가 아닌 실제 속한 분류가 어디인지 표시
sparse_target_dist = tf.constant([2])
sparse_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = unscaled_logits, labels= sparse_target_dist)
tf.print(sparse_xentropy)

# plt.subplot(2,1,1)
plt.plot(x_vals, l2_y_vals, 'b-', label = 'L2 Loss')
plt.plot(x_vals, l1_y_vals, 'r--', label = 'L1 Loss')
plt.plot(x_vals, phuber1_y_vals, 'k-.', label = 'P-Huber Loss(0.25)')
plt.plot(x_vals, phuber2_y_vals, 'g:', label = 'P-Huber Loss(0.5)')
plt.ylim(-0.2, 0.4)
plt.legend(loc = 'lower right', prop={'size':11})
plt.show()

plt.plot(x_vals, phuber1_y_vals, 'k:', label = 'P-Huber Loss(0.25)')
plt.plot(x_vals, huber1_y_vals, 'k-', label = 'Huber Loss(0.25)')
plt.plot(x_vals, phuber2_y_vals, 'g:', label = 'P-Huber Loss(0.5)')
plt.plot(x_vals, huber2_y_vals, 'g-', label = 'Huber Loss(0.5)')
plt.ylim(-0.2, 0.4)
plt.legend(loc = 'lower right', prop={'size':11})
plt.show()

plt.plot(x_vals2, hinge_y_vals, 'b-', label='Hinge Loss')
plt.plot(x_vals2, xentropy_y_vals, 'r--', label = 'Cross Entropy Loss')
plt.plot(x_vals2, xentropy_sigmoid_y_vals, 'k-.', label = 'Cross Entropy Sigmoid Loss')
plt.plot(x_vals2, xentropy_weighted_y_vals, 'g:', label='Weighted Cross Entropy Loss (x0.5)')
plt.ylim(-1.5, 3)
plt.legend(loc='lower right', prop={'size':11})
plt.show()