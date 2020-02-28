import tensorflow as tf
import os

def linear_regression():
  ## add name space
  with tf.variable_scope("prepare_data"):
  ## 1) generate dataset
    X = tf.random_normal(shape=[100,1],name="feature")
    y_true = tf.matmul(X,[[0.8]]) + 0.7

  ## add name space
  with tf.variable_scope("create_model"):
    ## 2) build model
    weights = tf.Variable(initial_value=tf.random_normal(shape=[1,1]),name="Weights")
    bias = tf.Variable(initial_value=tf.random_normal(shape=[1,1]),name="Bias")
    y_predict = tf.matmul(X, weights) + bias

  ## add name space
  with tf.variable_scope("loss_function"):
    ## 3) build loss function
    error = tf.reduce_mean(tf.square(y_predict - y_true))

  ## add name space
  with tf.variable_scope("optimizer"):
    ## 4) optimize loss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)

  ## 2_ collect variables
  tf.summary.scalar("error",error)
  tf.summary.histogram("weights",weights)
  tf.summary.histogram("bias",bias)
  ## 3_ merge variables
  merged = tf.summary.merge_all()

  ## create Saver object
  saver = tf.train.Saver()

  ## 5) initialize variables
  init = tf.global_variables_initializer()
  ## 6) open a session
  with tf.Session() as sess:
    sess.run(init)
    print("brefore: weights %f and bias %f, loss %f." % (weights.eval(), bias.eval(), error.eval()))
    ## 1_create event file
    file_writer = tf.summary.FileWriter('./temp/linear',graph=sess.graph)
    
    ## train the model
    # for i in range(1000):
    #   sess.run(optimizer)
    #   print("No. %d after: weights %f and bias %f, loss %f." % (i, weights.eval(), bias.eval(), error.eval()))
    #   ## 4_ run merginng variables
    #   summary = sess.run(merged)
    #   ## 5_ write on event file
    #   file_writer.add_summary(summary,i)
    #   ## save model
    #   if i % 10 == 0:
    #     saver.save(sess,"./temp/save/my_linear.ckpt")

    ## restore the model
    if os.path.exists("./temp/save/checkpoint"):
      saver.restore(sess,"./temp/save/my_linear.ckpt")
    print("after: weights %f and bias %f, loss %f." % (weights.eval(), bias.eval(), error.eval()))
  return None

if __name__ == "__main__":
  linear_regression()