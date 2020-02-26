import tensorflow as tf

def linear_regression():
  ## 1) generate dataset
  X = tf.random_normal(shape=[100,1])
  y_true = tf.matmul(X,[[0.8]]) + 0.7
  ## 2) build model
  weights = tf.Variable(initial_value=tf.random_normal(shape=[1,1]))
  bias = tf.Variable(initial_value=tf.random_normal(shape=[1,1]))
  y_predict = tf.matmul(X, weights) + bias
  ## 3) build loss function
  error = tf.reduce_mean(tf.square(y_predict - y_true))
  ## 4) optimize loss
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)
  ## 5) initialize variables
  init = tf.global_variables_initializer()
  ## 5) open a session
  with tf.Session() as sess:
    sess.run(init)
    print("brefore: weights %f and bias %f, loss %f." % (weights.eval(), bias.eval(), error.eval()))
    for i in range(1000):
      sess.run(optimizer)
      print("No. %d after: weights %f and bias %f, loss %f." % (i, weights.eval(), bias.eval(), error.eval()))
  return None

if __name__ == "__main__":
  linear_regression()