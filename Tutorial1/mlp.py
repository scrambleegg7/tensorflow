# -*- coding: utf-8 -*-


import tensorflow as tf 
import numpy as np

#import sklearn
from sklearn.datasets import make_moons

from sklearn.preprocessing import LabelBinarizer


import matplotlib.pyplot as plt 

def one_hot_labels(labels):
    return np.array([
        np.where(labels == 0, [1], [0]),
        np.where(labels == 1, [1], [0])
    ]).T
    

def randtheta(L_in,L_out):
    np.random.seed(0)
    rand_epsilon = np.sqrt(6) / np.sqrt(L_in+L_out)
    theta = (np.random.random((L_out, L_in + 1)) *(2*rand_epsilon)) - rand_epsilon
    return theta

def dataload():
        
    np.random.seed(0)
    X, y = make_moons(400, noise = 0.2)
        
    Train_YY = []  
    for i in y:       #making Train_Y a 2-D list
        if i == 1:
            Train_YY.append([1,0])
        else:
            Train_YY.append([0,1])

    y_ = np.array(Train_YY)

    return X,y_
    

def placeHolder(X_,y_):
    
    features = X_.shape[1]
    samples = y_.shape[0]
    
    X = tf.placeholder(tf.float32, [None, features])
    y = tf.placeholder(tf.float32, [None, 2])

    return X,y

def init_weights(shape):
    
    #return tf.Variable(tf.random_uniform(shape, -.01, .01))
    return tf.Variable(tf.random_normal(shape, stddev=0.8))

def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden): 
    # this network is the same as the previous one except with an extra hidden layer + dropout

    X = tf.nn.dropout(X, p_keep_input)
    h = tf.nn.relu(tf.matmul(X, w_h))

    h = tf.nn.dropout(h, p_keep_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2))

    h2 = tf.nn.dropout(h2, p_keep_hidden)
        
    return tf.matmul(h2, w_o)

def mlpModel():
    
    X,y = dataload()
    x_p, y_p = placeHolder(X,y)
    
    w1 = init_weights([2, 512]) # create symbolic variables
    w2 = init_weights([512, 512])
    w3 = init_weights([512, 2])


    p_keep_input = tf.placeholder(tf.float32)
    p_keep_hidden = tf.placeholder(tf.float32)
    
    H = model(x_p, w1, w2, w3, p_keep_input, p_keep_hidden)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(H, y_p))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    
    #train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)
    predict_op = tf.argmax(H, 1)
    
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.initialize_all_variables().run()

        for i in range(10000):
            
            costlist = []
            for start, end in zip(range(0, len(X), 50), range(50, len(X)+1, 50)):
        
                feed_dict={x_p: X[start:end], y_p: y[start:end],
                   p_keep_input: 0.8, p_keep_hidden: 0.5}
        
                e,a=sess.run([cost,train_op],feed_dict)
                costlist.append(e)
                #sess.run(train_op, feed_dict={x_p: X[start:end], y_p: y[start:end],
                #                              p_keep_input: 0.8, p_keep_hidden: 0.5}) 
            
            feed_dict={x_p: X, y_p: y,
                   p_keep_input: 1.0, p_keep_hidden: 1.0}
                    
            predict_ = sess.run(predict_op,feed_dict)
            answer = np.argmax(y,axis=1)
            accuracy = np.sum(predict_ == answer) / np.float(len(X))
            
            if i % 1000 == 0:                
                print "step : %d : cost : %s" % (i,np.mean(costlist))    
                print "-- accuracy : ", accuracy
                
                
def main():
    
    mlpModel()    



if __name__ == "__main__":
    main()
