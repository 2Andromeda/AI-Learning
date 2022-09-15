# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow as tf

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