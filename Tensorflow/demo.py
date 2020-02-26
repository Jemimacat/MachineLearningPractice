import tensorflow as tf

def tensorflow_demo():

  a = 1
  b = 2
  c = a + b
  print("The sum is " + str(c))

  a_t = tf.constant(2)
  b_t = tf.constant(3)
  c_t = a_t + b_t
  
  with tf.Session() as sess:
    c_t_value = sess.run(c_t)
    print(c_t_value)
    tf.summary.FileWriter('./',graph=sess.graph)
  return None


if __name__ == "__main__":
  tensorflow_demo()