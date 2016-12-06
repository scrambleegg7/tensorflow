# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


import matplotlib.pyplot as plt

def proc2():

    trX = np.linspace(-1, 1, 101)
    trY = 2 * trX + 3 + np.random.randn(*trX.shape) * 0.33

    
    print "trX trY shape : ", trX.shape, trY.shape    
    
    print "-- init placeholder --"
    X = tf.placeholder(tf.float32, [  trX.shape[0]  ])
    y = tf.placeholder(tf.float32, [  trY.shape[0]  ])
    
    W = tf.Variable([.0])
    b = tf.Variable([.0])

    y_Hypo = lineReg(X,W,b)


    cost = tf.reduce_mean(tf.square(y_Hypo - y))
    
    
    lam = 0.1
    train_step = tf.train.GradientDescentOptimizer(lam).minimize(cost)
    
    
    init = tf.initialize_all_variables()
    sess = tf.Session()    
    sess.run(init)

    hypos = []
    for i in range(1001):
        sess.run(train_step, feed_dict={X: trX, y: trY})
        if i % 100 == 0:
            print "%5d:(w,b)=(%10.4f, %10.4f)" % (i, sess.run(W), sess.run(b))
            hypo_  = sess.run(y_Hypo, feed_dict={X: trX, y: trY} )
            hypos.append(hypo_)

    #
    # Hypo 1 dimention - trY 1 dimention
    #
    correct_prediction = tf.sub(y_Hypo, y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) # [True, False, True, True] -> [1,0,1,1] -> 0.75.
    print "** actual accuracy",  accuracy.eval(session=sess, feed_dict={X: trX, y: trY} )

    w_ = sess.run(W)
    print w_, type(w_)

    b_ = sess.run(b)
    print b_, type(b_)    

    plt.plot(trX,trY,'o',c="r")
    plt.plot(trX,hypo_,'x',c="b")
    
    plt.show()


    
def lineReg(X, W,b):
    
    y = W * X + b
    return y
    #plt.plot(trX,trY,'o',c="r")
    #plt.show()


def main():
    proc2()


if __name__ == "__main__":
    main()