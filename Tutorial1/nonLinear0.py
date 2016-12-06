# -*- coding: utf-8 -*-


import tensorflow as tf 
import numpy as np

#import sklearn
from sklearn.datasets import make_moons

from sklearn.preprocessing import LabelBinarizer


import matplotlib.pyplot as plt 


def randtheta(L_in,L_out):
    np.random.seed(0)
    rand_epsilon = np.sqrt(6) / np.sqrt(L_in+L_out)
    theta = (np.random.random((L_out, L_in + 1)) *(2*rand_epsilon)) - rand_epsilon
    return theta

def dataload():
    
    x = np.array([])
    
    np.random.seed(0)
    X, y = make_moons(200, noise = 0.2)

    return X,y
    

def placeHolder(X_,y_):
    
    features = X_.shape[1]
    samples = y_.shape[0]
    
    X = tf.placeholder(tf.float32, [None, features])
    y = tf.placeholder(tf.float32, [samples, 2])


    return X,y


def cost():
    
    pass


def nnModel(X,Y):
        
    x_p, y_p = placeHolder(X,Y)
    
    input_dim = 2
    output_dim = 2

    number_hidden_nodes = 20


    W1 = tf.Variable(tf.random_normal([input_dim, number_hidden_nodes], stddev=0.01),
                      name="weights")
    b1 = tf.Variable(tf.zeros([1,number_hidden_nodes]), name="bias1")
    
    # place holder x w1    
    #a1 = tf.tanh(tf.add(tf.matmul(x_p,W1),b1))
    # relu     
    a1 = tf.nn.relu(tf.matmul(x_p,W1) + b1) # first layer.

    
    W2 = tf.Variable(tf.random_normal([number_hidden_nodes,output_dim]), name="weight2")
    b2 = tf.Variable(tf.zeros([1,output_dim]), name="bias2")
    
    a2 = tf.add(tf.matmul(a1, W2), b2)

    H = tf.nn.softmax(a2)
    #H = tf.nn.relu(a2)

    correct_prediction = tf.equal(tf.argmax(H,1), tf.argmax(y_p,1))
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #cross_entropy = -tf.reduce_sum(y_p*tf.log(H))
    
    cross_entropy = tf.reduce_mean(( (y_p * tf.log(H)) + 
        ((1 - y_p) * tf.log(1.0 - H)) ) * -1)
    
    
    lam = 0.01
    train_step = tf.train.GradientDescentOptimizer(lam).minimize(cross_entropy)
            
    """
    MUST start with session init before looping session run....
    Above values have to be initialized with default values.
    
    """    
    init = tf.initialize_all_variables()
    sess = tf.Session()    
    sess.run(init)
    
    print "-- convert to [xxx,1] for y"
    #Y = np.reshape( y, (y.shape[0],1) )
    X = np.reshape(X, (-1,2))
    
    for i in range(20000):
        # for (a,d) in zip(Train_X, Train_Y):
        e,a = sess.run([cross_entropy,train_step], feed_dict={x_p:X, y_p:Y})
        
        if i % 1000 == 0:
            print "-- Cost : ", e
            # print "Training cost=", training_cost, "W1=", W1.eval(), "b1=", b1.eval(),"W2=", W2.eval(), "b2=", b2.eval()
            # print output.eval({X:Train_X, Y:Train_YY})
            # print cross_entropy.eval({X:Train_X, Y:Train_YY})
            #print "Accuracy = ", accuracy.eval({x_p:X, y_p:Y}) 
                        
    #print('Hypothesis ', sess.run(H, feed_dict={x_p: X, y_p: Y}))
    
    Hypo = sess.run(H, feed_dict={x_p: X, y_p: Y})
    #predict = np.argmax(Hypo)
    predict = np.argmax(Hypo,axis=1)
    #print predict

    answer = np.argmax(Y,axis=1)
    #print answer
    
    print "-- accurancy -- ", np.sum(answer == predict) / np.float(len(Y))

    xx,yy = plotdata(X)
    z = np.c_[xx.ravel(), yy.ravel()] 
    print z
    Hypo_ = sess.run(H, feed_dict={x_p: z, y_p: Y})
    predict = np.argmax(Hypo_,axis=1)
    print predict
    Z = predict.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:,0],X[:,1],c=answer,cmap=plt.cm.Spectral)
    plt.show()

    

def plotdata(X):

    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
 
    
    return xx, yy

def LinerModel(X,y):


    x_p, y_p = placeHolder(X,y)


    number_hidden_nodes = 10
    W = tf.Variable(tf.random_uniform([2, number_hidden_nodes], -.01, .01))
    #b = tf.Variable(tf.random_uniform([number_hidden_nodes], -.01, .01))
    b = tf.Variable(tf.ones([number_hidden_nodes]))

    #
    #    input data is set into tf matmul 
    #
    hidden_1 = tf.nn.relu(tf.matmul(x_p,W) + b) # first layer.
    
    final_output = 1
    W2 = tf.Variable(tf.random_uniform([number_hidden_nodes,final_output], -.1, .1))

    #b2 = tf.Variable(tf.zeros([2]))
    Bias2 = tf.Variable(tf.zeros([1]))    
    hidden_2 = tf.matmul(hidden_1, W2)#+b2
    
    H = tf.sigmoid(hidden_2)
    
    
    cross_entropy = tf.reduce_mean(( (y_p * tf.log(H)) + 
        ((1 - y_p) * tf.log(1.0 - H)) ) * -1)

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
    
    
    print "-- convert to [xxx,1] for y"
    Y = np.reshape( y, (y.shape[0],1) )
    X = np.reshape(X, (-1,2))



    for step in range(50000):
        feed_dict={x_p: X, y_p: Y}
        #feed_dict={x: x_, y_:expect } # feed the net with our inputs and desired outputs.
        
        e,a=sess.run([cross_entropy,train_step],feed_dict)
        
        #if e<1:
        #    break # early stopping yay
        

        if step % 10000 == 0:        
            print "step : %d : cost : %s" % (step,e) # error/loss should decrease over time



def main():
    
    X,y = dataload()
    print X.shape,y.shape
    
    print "-- convert 2 byte for y"
    #lb = LabelBinarizer()
    #lb.fit(y)
    #y_ = lb.transform(y)
    Train_YY = []  
    for i in y:       #making Train_Y a 2-D list
        if i == 1:
            Train_YY.append([1,0])
        else:
            Train_YY.append([0,1])
    y_ = np.array(Train_YY)
    
    print "-- answer (binary)  \n", y_
    
    
    #LinerModel(X,y)
    nnModel(X,y_)
    


if __name__ == "__main__":
    main()
