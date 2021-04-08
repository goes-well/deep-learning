import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1' # 默认，显示所有信息 
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 warning 和 Error 
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3' # 只显示 Error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
print(tf.version.VERSION)
hello = tf.constant('Hello, TensorFlow!')
print(hello)
a = tf.constant([[1., 2.],[3., 4.]])
b = tf.constant([[5., 6.],[7., 8.]])
c = a+b
d = tf.matmul(a, b)
print(c)
print(d)

#自动求导
x = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:     # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
    y = tf.square(x)
y_grad = tape.gradient(y, x)        # 计算y关于x的导数
print(y, y_grad)
