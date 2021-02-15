import tensorflow as tf
import numpy as np

x1 = [73., 93., 89., 96., 73.] 
x2 = [80., 88., 91., 98., 66.]
x3 = [75., 93., 90., 100., 70.]
Y = [152., 185., 180., 196., 142.] 

w1 = tf.Variable(tf.random.normal([1]))
w2 = tf.Variable(tf.random.normal([1]))
w3 = tf.Variable(tf.random.normal([1]))
b = tf.Variable(tf.random.normal([1]))

learning_rate = 0.000001
for i in range(1000+1):
    with tf.GradientTape() as tape:
        hypothesis = w1 * x1 + w2 * x2 + w3 * x3 + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
    
    w1_grad, w2_grad, w3_grad, b_grad = tape.gradient(cost, [w1, w2, w3, b])
    
    w1.assign_sub(learning_rate * w1_grad)
    w2.assign_sub(learning_rate * w2_grad)
    w3.assign_sub(learning_rate * w3_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 50 == 0:
        print("{:5} | {:12.4f}".format(i, cost.numpy()))

# Using Matrix
print("use Matrix")
data = np.array([
    #x1, x2, x3, y
    [73, 80, 75, 152],
    [93, 88, 93, 185],
    [89, 91, 90, 180],
    [96, 98, 100, 196],
    [73, 66, 70, 142]
], dtype=np.float32)
# Slice data
# 행, 열
X = data[:, :-1] # 전체 로우, 마지막 컬럼 제외
Y = data[:, [-1]] # 전체 로우, 마지막 컬럼만
W = tf.Variable(tf.random.normal([3, 1]))
b = tf.Variable(tf.random.normal([1]))

for i in range(2001):
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean((tf.square(tf.matmul(X,W)+b - Y)))
    W_grad, b_grad = tape.gradient(cost, [W, b])
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)
    if i % 50 == 0:
        print("{:5} | {:12.4f}".format(i, cost.numpy()))
    