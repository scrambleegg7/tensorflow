# -*- coding: utf-8 -*


import tensorflow as tf 
import numpy as np

#from tensorflow.examples.tutorials.mnist import input_data



def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def weight_variable(shape):
    
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def placeHolder():
        
    """
    
    just provide place holder to build cupsles 
    any values are accepted.
    but, they are input with another step.
    
    """
    
    
    X = tf.placeholder(tf.float32, [None, 2])
    y = tf.placeholder(tf.float32, [4, 1])


    return X,y
    
    
    
def proc3():
    
    
     #sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})    
    
    
    x_placeH,y_placeH = placeHolder()
    
    number_hidden_nodes = 7
    W = tf.Variable(tf.random_uniform([2, number_hidden_nodes], -.01, .01))
    b = tf.Variable(tf.random_uniform([number_hidden_nodes], -.01, .01))

    #
    #    input data is set into tf matmul 
    #
    hidden_1 = tf.nn.relu(tf.matmul(x_placeH,W) + b) # first layer.
    
    final_output = 1
    W2 = tf.Variable(tf.random_uniform([number_hidden_nodes,final_output], -.1, .1))

    #b2 = tf.Variable(tf.zeros([2]))
    Bias2 = tf.Variable(tf.zeros([1]))    
    hidden_2 = tf.matmul(hidden_1, W2)#+b2
    
    H = tf.sigmoid(hidden_2)
    
    
    cross_entropy = tf.reduce_mean(( (y_placeH * tf.log(H)) + 
        ((1 - y_placeH) * tf.log(1.0 - H)) ) * -1)
        
        
    # Define loss and optimizer
    cross_entropy_ = -tf.reduce_sum(y_placeH * tf.log(H))
    
    
    lam = 0.1
    train_step = tf.train.GradientDescentOptimizer(lam).minimize(cross_entropy)
    #train_step = tf.train.GradientDescentOptimizer(lam).minimize(cost)

    #tf.initialize_all_variables().run()
    
    """
    MUST start with session init before looping session run....
    Above values have to be initialized with default values.
    
    
    """    
    init = tf.initialize_all_variables()
    sess = tf.Session()    
    sess.run(init)
    
    
    XOR_X = np.array([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]] )
    XOR_Y = np.array([0,1,1,0])
    
    
    #XOR_X = [[0,0],[0,1],[1,0],[1,1]]
    #XOR_Y = [[0],[1],[1],[0]]
     
    #print "-- shpae of training and labels : ", XOR_X.shape, XOR_Y.shape

    XOR_Y = np.reshape(XOR_Y,(4,1)  )  

    for step in range(50000):
        feed_dict={x_placeH: XOR_X, y_placeH: XOR_Y}
        #feed_dict={x: x_, y_:expect } # feed the net with our inputs and desired outputs.
        
        e,a=sess.run([cross_entropy,train_step],feed_dict)
        
        #if e<1:
        #    break # early stopping yay
        

        if step % 10000 == 0:        
            print "step : %d : cost : %s" % (step,e) # error/loss should decrease over time
            print('Hypothesis ', sess.run(H, feed_dict={x_placeH: XOR_X, y_placeH: XOR_Y}))
            
            #print "W2 : ", sess.run(W2)
    
    
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(H,1), tf.argmax(y_placeH,1)) # argmax along dim-1
    print  "-- correct predict : ",correct_prediction    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) # [True, False, True, True] -> [1,0,1,1] -> 0.75.

    print "-- final accuracy of model %s"% (accuracy.eval(session=sess, feed_dict = {x_placeH: XOR_X, y_placeH:XOR_Y} ) )

    #print("test accuracy %g" % accuracy.eval(session=sess, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


    learned_output=tf.argmax(H,1)
    #print learned_output.eval( session=sess)
    print learned_output.eval( session=sess, feed_dict= {x_placeH: XOR_X})

    
    

def proc2():
    
    x_,y_ = placeHolder()
    
    
    """
    this is an actual testing / labeling data to be passed into variables.
    """
    training_data = np.array([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]] )
    training_labels = np.array([0,1,1,0])
    
    # set hidden number
    hidden_numbers = 20
    W = tf.Variable(tf.zeros([4, hidden_numbers]))
    
    
    """
    
    build appropriate neuro model... conv2d etc... any styles are written 
    in order to get right answer very close to testing label.
    
    """


    W = tf.Variable(tf.random_uniform([2, number_hidden_nodes], -.01, .01))
    b = tf.Variable(tf.random_uniform([number_hidden_nodes], -.01, .01))
    hidden  = tf.nn.relu(tf.matmul(x,W) + b) # first layer.

    # the XOR function is the first nontrivial function, 
    # for which a two layer network is needed.
    W2 = tf.Variable(tf.random_uniform([number_hidden_nodes,2], -.1, .1))
    b2 = tf.Variable(tf.zeros([2]))
    hidden2 = tf.matmul(hidden, W2)#+b2
    
    y = tf.nn.softmax(hidden2)    


    

    print "-- shpae of training and labels : ", training_data.shape, training_labels.shape    
    
    #training_data = []
    #training_labels = []
    with tf.Session() as sess:
        data_initializer = tf.placeholder(dtype=training_data.dtype,
                                    shape=training_data.shape)
        label_initializer = tf.placeholder(dtype=training_labels.dtype,
                                     shape=training_labels.shape)
                                     
        input_data = tf.Variable(data_initializer, trainable=False, collections=[])
        input_labels = tf.Variable(label_initializer, trainable=False, collections=[])

        sess.run(input_data.initializer,
           feed_dict={data_initializer: training_data})
        sess.run(input_labels.initializer,
           feed_dict={label_initializer: training_labels})


def main():
    
    proc3()
    
    


if __name__ == "__main__":
    main()
    
    