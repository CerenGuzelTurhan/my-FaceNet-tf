"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import facenet
import math



def extractor(data_dir, model = './models/facenets/20170512-110547', lfw_batch_size = 100, image_size = 160, num_epochs = 20, lfw_pairs= './data/pairs.txt', min_images_per_labels = 3, is_train = True):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            # Load the model
            facenet.load_model(model)

            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
                
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            train_set, test_set = facenet._get_test_and_train_set(data_dir, min_num_images_per_label=min_images_per_labels, 
                                                      split_ratio=0.7)

            train_images, train_labels, class_names = facenet._load_images_and_labels(train_set, image_size=160, batch_size=lfw_batch_size,
                                                                  num_epochs=num_epochs,
                                                                  random_flip=True, random_brightness=True, num_threads = 8,
                                                                  random_contrast=True)
           
            test_images, test_labels, test_class_names = facenet._load_images_and_labels(test_set, image_size=160, batch_size=lfw_batch_size,
                                                                  num_epochs=num_epochs,
                                                                  random_flip=True, random_brightness=True,num_threads = 8,
                                                                  random_contrast=True)
            emb_train, label_train = get_embeddings(sess, train_images, train_labels, train_set, lfw_batch_size, images_placeholder, embeddings, phase_train_placeholder)
            emb_test, label_test = get_embeddings(sess, test_images, test_labels, test_set, lfw_batch_size, images_placeholder, embeddings, phase_train_placeholder)
            
        
    return [emb_train, emb_test, label_train, label_test]
            
def get_embeddings(sess, images, labels, imageset, batch_size, images_placeholder, embeddings, phase_train_placeholder):
      
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        
    emb_array = None
    label_array = None

  
    nrof_images = len(imageset)
    nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
    for i in range(nrof_batches):
        batch_images, batch_labels  = sess.run([images, labels])     
        emb = sess.run(embeddings,feed_dict={images_placeholder: batch_images, phase_train_placeholder: False})

        emb_array = np.concatenate([emb_array, emb]) if emb_array is not None else emb
        label_array = np.concatenate([label_array, batch_labels]) if label_array is not None else batch_labels
             
      
    coord.request_stop()
    coord.join(threads=threads)
    return emb_array, label_array
