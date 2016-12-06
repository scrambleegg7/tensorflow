# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelBinarizer

from caffeBase.myDirClassNocv2 import myDirClass

import numpy as np
import os
import platform as pl


def readDemo():
    
    if pl.system() == "Linux":
        udir = "/home/hideaki/SynologyNFS/myProgram/pythonProj/KaggleData/Alzheimer"
    else:    
        udir = "/Users/donchan/Documents/myData/KaggleData/Alzheimer"    
    filename = "adni_demographic_master_kaggle.csv"
    fullpath = os.path.join(udir,filename)
    
    demo = pd.read_csv(fullpath)
    print "-- demographic data : \n", demo.head()
    

    trX_subjs = demo[(demo['train_valid_test']==0)]
    trY_diagnosis = np.asarray(trX_subjs.diagnosis)

    vaX_subjs = demo[(demo['train_valid_test']==1)]
    vaY_diagnosis = np.asarray(vaX_subjs.diagnosis)

    train_orig = trY_diagnosis
    valid_orig = vaY_diagnosis    
    
    images_per_sub = 62

    """
    diagnosis x 62 == total images of one Subjects    
    
    """
    trY_all = []
    for n in trY_diagnosis:
        for i in range(images_per_sub):
            trY_all.append(n)

    trY_all = np.asarray(trY_all)
    
    vaY_all = []
    for n in vaY_diagnosis:
        for i in range(images_per_sub):
            vaY_all.append(n)
    vaY_all = np.asarray(vaY_all)    

    trainingOneHot =LabelBinarizer().fit_transform(trY_all)
    
    validOneHot =LabelBinarizer().fit_transform(vaY_all)
    print "-- length for traing / valid target flag",  len(trainingOneHot), len(validOneHot)
    
    return trainingOneHot,validOneHot,train_orig,valid_orig

def StackImage(files):

    prev_data = np.array([])
    s_image = np.array([])
    
    for idx, f in enumerate(files):
        print "- reading...", f
        img_data = np.load(f)
        
        if idx > 0:
            s_image = np.vstack((prev_data,img_data))
            prev_data = s_image
        else:
            prev_data = img_data
                    
    return s_image    

def alzDataDir():
    
    training_idx = range(1,4)

    myDirCls = myDirClass()
    
    files = []    
    for idx in training_idx:
        print ".... reading imgset_%d : dirctory" % idx
        

        if pl.system() == "Linux":
            udir = "/home/hideaki/SynologyNFS/myProgram/pythonProj/KaggleData/Alzheimer/imgset_%d" % idx
        else:
            udir = "/Users/donchan/Documents/myData/KaggleData/Alzheimer/imgset_%d" % idx
        myDirCls.getFiles(udir) 
        trainings = [f for f in myDirCls.getFileList() if f[-3:] == 'npy' ]
        
        files.extend(trainings)

    return files

def next_batch(data, label, batch_size=100):
    perm = np.arange(data.shape[0])
    r = np.random.permutation(perm)
    return data[r][:batch_size], label[r][:batch_size]

def convTest(images,diagnosis):
    
    print "-- build place holder ... "
    x_p = tf.placeholder(tf.float32, [None, 96,96])
    y_p = tf.placeholder(tf.float32, [None, 3])
    
    
    #sess.run(init)
    #sess.run(tf.initialize_all_variables())
    
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    
    x_image = tf.reshape(x_p, [-1,96,96,1])
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(5):
            batch_train_x, batch_train_y = next_batch(images, diagnosis, 100)
            
            feed_dict={x_p: batch_train_x, y_p: batch_train_y}
            h_conv1_res = sess.run(h_conv1,feed_dict)
            print "h_conv1 shape after regularization:",i,h_conv1_res.shape            
            print "--step--:",i
        
    

def convProc(images,diagnosis):
    
    print "-- build place holder ... "
    x = tf.placeholder(tf.float32, [None, 96,96])
    y_ = tf.placeholder(tf.float32, [None, 3])
    
    
    print "-- build CNN model for images ..."
    #W = tf.Variable(tf.zeros([ 96, 96, 10]))
    #b = tf.Variable(tf.zeros([10]))
    #y = tf.nn.softmax(tf.matmul(x, W) + b)
    
    sess = tf.Session()
    #sess.run(init)
    #sess.run(tf.initialize_all_variables())
    
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    
    x_image = tf.reshape(x, [-1,96,96,1])
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 3])
    b_fc2 = bias_variable([3])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    #correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    sess.run(tf.initialize_all_variables())

    for i in range(200):
        #batch = mnist.train.next_batch(50)
        # batch size = 6200 NOT fit to tensorflow
        data, label = next_batch(images,diagnosis,124  )
    
        if i%100 == 0:
            #train_accuracy = accuracy.eval(session=sess, feed_dict={x:data, y_: label, keep_prob: 1.0})
    
            #print("step %d, training accuracy %g"%(i, train_accuracy))
            #if train_accuracy > 0.98:
            print "--  step : %d" % i
        
        train_step.run(session=sess, feed_dict={x: data, y_: label, keep_prob: 0.5})

    
    #print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))




def weight_variable(shape):
    
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    

def main():
    print "test"
    
    trainingOneHot,validOneHot,train_orig,valid_orig = readDemo()

    files = alzDataDir()    
    train_image = StackImage(files)
    
    print "-- reading trainging image shape ...............",train_image.shape
    training_size = train_image.shape[0]
    
    data, label = next_batch(train_image,trainingOneHot[:training_size],6200   )
    print " -- batch - image and label shape ....", data.shape, label.shape
    diagnosis = trainingOneHot[:training_size]
    
    convTest(train_image,diagnosis)
    
        

if __name__ == "__main__":
    main()