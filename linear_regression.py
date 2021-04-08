import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1' # 默认，显示所有信息 
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 warning 和 Error 
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3' # 只显示 Error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
print(tf.version.VERSION)
import numpy as np

X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
Y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())  #归一化
Y = (Y_raw - Y_raw.min()) / (Y_raw.max() - Y_raw.min()) #归一化
print(X)
print(Y)
X = tf.constant(X)
Y = tf.constant(Y)
print(X)
print(Y)
a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
variables = [a, b]

num_epoch = 50000
optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4)
for e in range(num_epoch):
    # 使用tf.GradientTape()记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = tf.reduce_sum(tf.square(y_pred - Y))
    # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
    grads = tape.gradient(loss, variables)
    # TensorFlow自动根据梯度更新参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

print(variables)
x_target=(2016-X_raw.min()) / (X_raw.max() - X_raw.min())
pred_target=a*x_target + b
y_target=tf.constant(pred_target*(Y_raw.max() - Y_raw.min()) + Y_raw.min())
print(x_target)
print(pred_target)
print(y_target)


x_target=(2018-X_raw.min()) / (X_raw.max() - X_raw.min())
pred_target=a*x_target + b
y_target=tf.constant(pred_target*(Y_raw.max() - Y_raw.min()) + Y_raw.min())
print(x_target)
print(pred_target)
print(y_target)
