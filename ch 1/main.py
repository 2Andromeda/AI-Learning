# This is a sample tensorflow coding for devduck

import sys
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow.nn as nn

import pandas as pd
import numpy as np
import requests
import io
import tarfile

from sklearn import datasets
from zipfile import ZipFile

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

print(sess.run(nn.relu([-3., 3., 10.]))) #ReLU(Rectified Linear Unit) f(x) = max(0, x)
print(sess.run(nn.relu6([-3., 3., 10.]))) #ReLU6(선형 증가량에 상한 설정됨) f(x) = max((0, x), 6)
print(sess.run(nn.sigmoid([-1., 0., 1.]))) #sigmoid 함수 f(x) = 1/(1+exp(-x))
print(sess.run(nn.tanh([-1., 0., 1.]))) #tanh 함수 f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
print(sess.run(nn.softsign([-1., 0., 1.]))) #softsign 함수 f(x) = x/(abs(x) +1)
print(sess.run(nn.softplus([-1., 0., 1.]))) #softplus 함수 f(x) = log(exp(x) + 1) 매끄럽게 만든 ReLU 함수
print(sess.run(nn.elu([-1., 0., 1.]))) #ELU(Exponential lindear unit) 함수 f(x) = exp(x)+1 (x<0), x (x>=0)

sess.close()

#붓꽃 데이터
iris=datasets.load_iris()
print(len(iris.data))
print(len(iris.target))
print(iris.data[0])
print(set(iris.target))

#신생아 체중 데이터
birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat'
birth_file = requests.get(birthdata_url)
birth_data = birth_file.text.split('\r\n')
birth_header = birth_data[0].split('\t')
birth_data = [[float(x) for x in y.split('\t') if len(x) >= 1] for y in birth_data[1:] if len(y)>=1]
print(len(birth_data))
print(len(birth_data[0]))

#캘리포니아 주택 데이터
housing_data = datasets.fetch_california_housing()
print(len(housing_data.data))
print(len(housing_data.data[0]))

#MNIST 필기 데이터 (Mixed National Institute of Standards and Technology)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
print(len(x_train))
print(len(x_test))
print(len(y_train))

#스팸-비스팸 문자 데이터
zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
r= requests.get(zip_url) #get으로 파일 받아내기
z = ZipFile(io.BytesIO(r.content)) #zip파일 읽기
file = z.read('SMSSpamCollection')
text_data = file.decode()
text_data=text_data.encode('ascii', errors = 'ignore')
text_data = text_data.decode().split('\n')
text_data = [x.split('\t') for x in text_data if len(x) >=1]
[text_data_target, text_data_train] =[list(x) for x in zip(*text_data)]
print(len(text_data_train))
print(set(text_data_target))
print(text_data_train[1])

#영화 리뷰 데이터
movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
r = requests.get(movie_data_url)
#데이터를 임시 객체로 스트리밍
stream_data = io.BytesIO(r.content)
tmp = io.BytesIO()
while True:
    s = stream_data.read(16384)
    if not s:
        break
    tmp.write(s)
stream_data.close()
tmp.seek(0)
#tar 파일 풀기
tar_file = tarfile.open(fileobj=tmp, mode="r:gz")
pos = tar_file.extractfile('rt-polaritydata/rt-polarity.pos')
neg = tar_file.extractfile('rt-polaritydata/rt-polarity.neg')
#긍정/부정 리뷰 저장(인코딩도 함께 처리)
pos_data = []
for line in pos:
    pos_data.append(line.decode('ISO-8859-1').encode('ascii', errors='ignore').decode())
neg_data = []
for line in neg:
    neg_data.append(line.decode('ISO-8859-1').encode('ascii', errors='ignore').decode())
tar_file.close()
print(len(pos_data))
print(len(neg_data))
print(neg_data[0])

#tensorflow v2 에서 적용되는 방법 - 아직 작동 미완성
print(tf.divide(3, 4))
print(tf.multiply(3, 4))
tf.print(tf.truediv(3, 4), output_stream=sys.stderr)
