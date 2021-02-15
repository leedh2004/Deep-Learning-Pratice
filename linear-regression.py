import tensorflow as tf

x_data = [1,2,3,4,5]
y_data = [1,2,3,4,5]

W = tf.Variable(2.9) 
b = tf.Variable(0.5) 

learning_rate =0.01

for i in range(100+1):
    # Gradient Descent, Tape에 기록
    with tf.GradientTape() as tape:
        hypothesis = W * x_data + b
        # reduce_mean은 차원이 하나 줄어듬.
        # 차원이 하나 줄어들면서 평균을 냄
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))

    # W와 b에 대한 기울기(미분 값)
    W_grad, b_grad = tape.gradient(cost, [W, b])
    # A.assign_sub(B) == (A = A - b)
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)
    if i % 10 == 0:
        print("{:5}|{:10.4f}|{:10.4}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))

# Convex function ==> local minimum이 global minimum인 것