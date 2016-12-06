# -*- coding: utf-8 -*-


import tensorflow as tf

import numpy as np

#import sklearn
from sklearn.datasets import make_moons

import matplotlib.pyplot as plt 

def plotdata(X):

    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
     
    return xx, yy

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
        
    y_ = one_hot_labels(y)

    return X,y_
    

def placeHolder(X_,y_):
    
    features = X_.shape[1]
    out = y_.shape[1]
    
    X = tf.placeholder(tf.float32, [None, features])
    y = tf.placeholder(tf.float32, [None, out])

    return X,y
    
def next_batch(data, label, batch_size):
    perm = np.arange(data.shape[0])
    np.random.shuffle(perm)
    return data[perm][:batch_size], label[perm][:batch_size]

def init_weights(shape):
    
    #return tf.Variable(tf.random_uniform(shape, -.01, .01))
    return tf.Variable(tf.random_normal(shape, stddev=0.5))
    #return tf.Variable(tf.truncated_normal(shape, stddev=0.5))

def model(x,W1,W2,W3,W4):
    
    b1 = tf.Variable(tf.constant(0.0, shape=[10], name='bias1'))
    h1 = tf.nn.relu(tf.matmul(x,W1) + b1)

    b2 = tf.Variable(tf.constant(0.0, shape=[20], name='bias2'))
    h2 = tf.nn.relu(tf.matmul(h1,W2) + b2)

    b3 = tf.Variable(tf.constant(0.0, shape=[10], name='bias3'))
    h3 = tf.nn.relu(tf.matmul(h2,W3) + b3)

    b4 = tf.Variable(tf.constant(0.0, shape=[2], name='bias4'))
    H = tf.nn.softmax(tf.matmul(h3,W4) + b4)

    return H
    
def nnSamples():
    
    X,y = dataload()
    x_p, y_p = placeHolder(X,y)
    
    W1 = init_weights([2,10])
    W2 = init_weights([10,20])
    W3 = init_weights([20,10])
    W4 = init_weights([10,2])
    
    H = model(x_p,W1,W2,W3,W4)
    
    predict_op = tf.argmax(H, 1)
    
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_p * tf.log(H), reduction_indices=[1]))
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
        
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(10000):
            
            batch_size = 100
            
            batch_train_x, batch_train_y = next_batch(X, y, batch_size)
            
            feed_dict={x_p: batch_train_x, y_p: batch_train_y}
            e,a = sess.run([cross_entropy,train_op], feed_dict)
            
            if i % 1000 == 0:
                print "step : %d : cost : %s" % (i,e)
            
        
        
        feed_dict={x_p: X, y_p: y}
                    
        predict_ = sess.run(predict_op,feed_dict)
        answer = np.argmax(y,axis=1)
        accuracy = np.sum(predict_ == answer) / np.float(len(X))
        
        print "-- accuracy : ", accuracy
        #print "-- answer:", answer
        #print "-- predict:", predict_

        xx,yy = plotdata(X)
        z = np.c_[xx.ravel(), yy.ravel()] 

        feed_dict={x_p: z, y_p: y}
        #Hypo_ = sess.run(H, feed_dict={x_p: z, y_p: y})
        predict = sess.run(predict_op,feed_dict)
                
        Z = predict.reshape(xx.shape)
    
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(X[:,0],X[:,1],c=answer,cmap=plt.cm.Spectral)
        plt.show()

            
            
def main():
    nnSamples()


if __name__ == "__main__":
    main()
    
    
