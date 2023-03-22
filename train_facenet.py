import tensorflow as tf
import numpy as np
import itertools
import time
import os
from datetime import datetime
from tensorflow.python.platform import gfile

import dataio as d
import facenet
import importlib

class trainfacenet():
    def __init__(self, images,
                     labels, 
                     target_names,
                     batch_size=100,
                     image_size = 182,
                     max_epoch = 500, 
                     epoch_size=1000, 
                     optimizer = 'ADAGRAD', 
                     keep_probability = 1.0, 
                     weight_decay = 0.0,
                     learning_rate_decay_factor = 1.0, 
                     learning_rate_decay_epochs= 100,
                     moving_average_decay = 0.9999,
                     embedding_size = 128, 
                     gpu_memory_fraction = 1.0, 
                     alpha = 0.2, 
                     pretrained_model = './models/facenets/20170512-110547',
                     model_name = 'models.inception_resnet_v1', 
                     model_dir = './models/facenets', 
                     log_dir = './logs/facenets', 
                     dataset_dir =  './dataset/lfw',
                     pair_file =  './data/pairs.txt',
                     people_per_batch = 45,
                     images_per_person = 40,
                     learning_rate_schedule_file='./data/learning_rate_schedule.txt'):
        
        self.images = images
        self.labels = labels 
        self.target_names = target_names
        self.batch_size = batch_size
        self.image_size = image_size
        self.max_epoch = max_epoch 
        self.epoch_size = epoch_size 
        self.optimizer = optimizer 
        self.keep_probability = keep_probability, 
        self.weight_decay = weight_decay,
        self.learning_rate_decay_factor =  learning_rate_decay_factor
        self.learning_rate_decay_epochs = learning_rate_decay_epochs
        self.moving_average_decay = moving_average_decay
        self.embedding_size = embedding_size
        self.gpu_memory_fraction = gpu_memory_fraction
        self.alpha = alpha
        self.pretrained_model = pretrained_model
        self.model_name = model_name
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.dataset_dir = dataset_dir
        self.pair_file = pair_file
        self.people_per_batch = people_per_batch
        self.images_per_person = images_per_person
        self.learning_rate_schedule_file = learning_rate_schedule_file
        
        network = importlib.import_module(model_name)
        self.subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        self.log_dir = os.path.join(os.path.expanduser(log_dir), self.subdir)
        if not os.path.isdir(self.log_dir):  # Create the log directory if it doesn't exist
            os.makedirs(self.log_dir)
        self.model_dir = os.path.join(os.path.expanduser(model_dir), self.subdir)
        if not os.path.isdir(self.model_dir):  # Create the model directory if it doesn't exist
            os.makedirs(self.model_dir)
    
        if pair_file:
            self.pairs = d.read_pairs(self.pair_file)
            self.pairpaths, self.issame_list = d.get_pairpaths(self.dataset_dir, self.pairs)
            self.pair_images, self.pair_labels = d.get_pairs(self.pairs, self.images, self.labels, self.target_names)
        
        with tf.Graph().as_default(): 
            self.global_step = tf.Variable(0, trainable=False)
            
            self.learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
            self.batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
            self.phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')  
            self.image_batch = tf.placeholder(tf.float32, shape=(None,image_size, image_size, 3), name='images')
            self.label_batch = tf.placeholder(tf.int64, shape=(None,1), name='labels')
             
            prelogits, _ = network.inference(self.image_batch, keep_probability, 
                phase_train=self.phase_train_placeholder, bottleneck_layer_size=embedding_size, 
                weight_decay=weight_decay)
    
            self.embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
            self.anchor, self.positive, self.negative = tf.unstack(tf.reshape(self.embeddings, [-1,3,embedding_size]), 3, 1)
            self.triplet_loss = facenet.triplet_loss(self.anchor, self.positive, self.negative, self.alpha)
            
            self.learning_rate = tf.train.exponential_decay(self.learning_rate_placeholder, self.global_step, learning_rate_decay_epochs*epoch_size, learning_rate_decay_factor, staircase=True)
            tf.summary.scalar('learning_rate', self.learning_rate)
    
            # Calculate the total losses
            self.regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.total_loss = tf.add_n([self.triplet_loss] + self.regularization_losses, name='total_loss')
            
            # Build a Graph that trains the model with one batch of examples and updates the model parameters
            self.train_op = facenet.train(self.total_loss, self.global_step, self.optimizer, self.learning_rate, self.moving_average_decay, tf.global_variables())
        
            # Create a saver
            self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
    
            # Build the summary operation based on the TF collection of Summaries.
            self.summary_op = tf.summary.merge_all()
    
            # Start running operations on the Graph.
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))        
    
            # Initialize variables
            self.sess.run(tf.global_variables_initializer(), feed_dict={self.phase_train_placeholder:True})
            self.sess.run(tf.local_variables_initializer(), feed_dict={self.phase_train_placeholder:True})
    
            self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
            self.coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=self.coord, sess=self.sess)
            
            with self.sess.as_default():
    
                if pretrained_model:
                    print('Restoring pretrained model: %s' % pretrained_model)
                    model_exp = os.path.expanduser(pretrained_model)
                    if (os.path.isfile(model_exp)):
                        print('Model filename: %s' % model_exp)
                        with gfile.FastGFile(model_exp,'rb') as f:
                            graph_def = tf.GraphDef()
                            graph_def.ParseFromString(f.read())
                            tf.import_graph_def(graph_def, name='')
                    else:
                        print('Model directory: %s' % model_exp)
                        meta_file, ckpt_file = facenet.get_model_filenames(model_exp)
        
                        print('Metagraph file: %s' % meta_file)
                        print('Checkpoint file: %s' % ckpt_file)
      
                    self.saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
                    self.saver.restore(self.sess, os.path.join(model_exp, ckpt_file))
     
                # Training and validation loop
                epoch = 0
                while epoch < max_epoch:
                    step = self.sess.run(self.global_step, feed_dict=None)
                    epoch = step // epoch_size
                    # Train for one epoch
                    self.train(epoch)
    
                    # Save variables and the metagraph if it doesn't exist already
                    self.save_variables_and_metagraph(step)
    
                    self.evaluate(step)
        return self.model_dir

    def train(self, epoch):
        batch_number = 0
        nrof_examples = self.people_per_batch * self.images_per_person
        while batch_number < self.epoch_size:
            sampledimages, sampledlabels, num_per_class = self.sample_people(self.images, self.labels)
            start_time = time.time()
            labels_array = np.reshape(np.arange(nrof_examples),(-1,3))
            image_array = np.reshape(np.expand_dims(np.array(sampledimages),1), (-1,3))
            batch_images, batch_labels = self.sess.run([self.image_batch, self.label_batch], {self.image_batch: image_array, self.label_batch: labels_array})
            emb_array = np.zeros((nrof_examples, self.embedding_size))
            nrof_batches =  int(np.ceil(nrof_examples / self.batch_size))
            for i in range(nrof_batches):
                batch_size = min(nrof_examples-i*self.batch_size, self.batch_size)
                emb, lab = self.sess.run([self.embeddings, self.label_batch], feed_dict={self.batch_size_placeholder: batch_size, self.phase_train_placeholder: True})
                emb_array[lab,:] = emb
                 # Select triplets based on the embeddings
                 
                 
            print('Selecting suitable triplets for training')
            triplets, nrof_random_negs, nrof_triplets = self.select_triplets(emb_array, num_per_class, sampledimages)
            selection_time = time.time() - start_time
            # Perform training on the selected triplets
            nrof_batches = int(np.ceil(nrof_triplets*3/batch_size))
            triplet_images = list(itertools.chain(*triplets))
            labels_array = np.reshape(np.arange(len(triplet_images)),(-1,3))
            triplet_array = np.reshape(np.expand_dims(np.array(triplet_images),1), (-1,3))
            self.sess.run([self.image_batch, self.label_batch], feed_dict={self.image_batch: triplet_array, self.label_batch: labels_array})
            
            train_time = 0
            i = 0
            emb_array = np.zeros((nrof_examples, self.embedding_size))
            loss_array = np.zeros((nrof_triplets,))
            while i < nrof_batches:
                start_time = time.time()
                batch_size = min(nrof_examples-i*batch_size, batch_size)
                feed_dict = {self.batch_size_placeholder: batch_size, self.learning_rate_placeholder: self.learning_rate, self.phase_train_placeholder: True}
                err, _, step, emb, lab = self.sess.run([self.loss, self.train_op, self.global_step, self.embeddings, self.label_batch], feed_dict=feed_dict)
                emb_array[lab,:] = emb
                loss_array[i] = err
                duration = time.time() - start_time
                print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
                      (epoch, batch_number+1, self.epoch_size, duration, err))
                batch_number += 1
                i += 1
                train_time += duration
                
            # Add validation loss and accuracy to summary
            self.summary = tf.Summary()
            #pylint: disable=maybe-no-member
            self.summary.value.add(tag='time/selection', simple_value=selection_time)
            self.summary_writer.add_summary(self.summary, step)
        return step
            
    def select_triplets(self, embeddings, nrof_images_per_class, sampledimages):
        """ Select the triplets for training
        """
        trip_idx = 0
        emb_start_idx = 0
        num_trips = 0
        triplets = []
    
        for i in range(self.people_per_batch):
            nrof_images = int(nrof_images_per_class[i])
            for j in range(1,nrof_images):
                a_idx = emb_start_idx + j - 1
                neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
                for pair in range(j, nrof_images): # For every possible positive pair.
                    p_idx = emb_start_idx + pair
                    pos_dist_sqr = np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                    neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                    #all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                    all_neg = np.where(neg_dists_sqr-pos_dist_sqr<self.alpha)[0] # VGG Face selecction
                    nrof_random_negs = all_neg.shape[0]
                    if nrof_random_negs>0:
                        rnd_idx = np.random.randint(nrof_random_negs)
                        n_idx = all_neg[rnd_idx]
                        triplets.append((sampledimages[a_idx], sampledimages[p_idx], sampledimages[n_idx]))
                        #print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' % 
                        #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                        trip_idx += 1
    
                    num_trips += 1
    
            emb_start_idx += nrof_images
    
        np.random.shuffle(triplets)
        return triplets, num_trips, len(triplets)  

    def sample_people(self, images, labels):
        nrof_images = self.people_per_batch * self.images_per_person
      
        # Sample classes from the dataset
        nrof_classes = len(np.unique(labels))
        class_indices = np.arange(nrof_classes)
        np.random.shuffle(class_indices)
        
        i = 0
        sampled_images = []
        num_per_class = []
        sampled_labels = []
        # Sample images from these classes until we have enough
        while len(sampled_images)<nrof_images:
            class_index = class_indices[i]
            nrof_images_in_class = len(np.where(labels == class_index))
            image_indices = np.arange(nrof_images_in_class)
            np.random.shuffle(image_indices)
            nrof_images_from_class = min(nrof_images_in_class, self.images_per_person, nrof_images-len(sampled_images))
            idx = image_indices[0:nrof_images_from_class]
            class_all_images = images[np.where(labels == class_index)[0]]
            class_sampled_images = [class_all_images[j] for j in idx]
            sampled_labels += [class_index]*nrof_images_from_class
            sampled_images += class_sampled_images
            num_per_class.append(nrof_images_from_class)
            i+=1
        return sampled_images, sampled_labels, num_per_class

    def evaluate(self, step):
        start_time = time.time()
        # Run forward pass to calculate embeddings
        print('Running forward pass on LFW images: ', end='')
        nrof_images = len(self.issame_list)*2
        assert(len(self.pair_images)==nrof_images)
        labels_array = np.reshape(np.arange(nrof_images),(-1,3))
        image_array = np.reshape(np.expand_dims(np.array(self.pair_images),1), (-1,3))
        self.sess.run([self.image_batch, self.label_batch], feed_dict={self.image_batch: image_array, self.label_batch: labels_array})
        emb_array = np.zeros((nrof_images, self.embedding_size))
        nrof_batches = int(np.ceil(nrof_images / self.batch_size))
        label_check_array = np.zeros((nrof_images,))
        for i in range(nrof_batches):
            batch_size = min(nrof_images-i*self.batch_size, self.batch_size)
            emb, lab = self.sess.run([self.embeddings, self.label_batch], feed_dict={self.batch_size_placeholder: batch_size, self.learning_rate_placeholder: 0.0, self.phase_train_placeholder: False})
            emb_array[lab,:] = emb
            label_check_array[lab] = 1
        print('%.3f' % (time.time()-start_time))
        
        assert(np.all(label_check_array==1))
        
        _, _, accuracy, val, val_std, far = self.evaluate_acc(emb_array, self.issame_list, nrof_folds=10)
        
        print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
        print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
        lfw_time = time.time() - start_time
        # Add validation loss and accuracy to summary
        self.summary = tf.Summary()
        #pylint: disable=maybe-no-member
        self.summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
        self.summary.value.add(tag='lfw/val_rate', simple_value=val)
        self.summary.value.add(tag='time/lfw', simple_value=lfw_time)
        self.summary_writer.add_summary(self.summary, step)
        with open(os.path.join(self.log_dir,'lfw_result.txt'),'at') as f:
            f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))
            
    def evaluate_acc(self, embeddings, actual_issame, nrof_folds=10):
        # Calculate evaluation metrics
        thresholds = np.arange(0, 4, 0.01)
        embeddings1 = embeddings[0::2]
        embeddings2 = embeddings[1::2]
        tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
            np.asarray(actual_issame), nrof_folds=nrof_folds)
        thresholds = np.arange(0, 4, 0.001)
        val, val_std, far = facenet.calculate_val(thresholds, embeddings1, embeddings2,
            np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
        return tpr, fpr, accuracy, val, val_std, far
    
    
    def save_variables_and_metagraph(self,step):
        # Save the model checkpoint
        print('Saving variables')
        start_time = time.time()
        checkpoint_path = os.path.join(self.model_dir, 'model-%s.ckpt' % self.subdir)
        self.saver.save(self.sess, checkpoint_path, global_step=step, write_meta_graph=False)
        save_time_variables = time.time() - start_time
        print('Variables saved in %.2f seconds' % save_time_variables)
        metagraph_filename = os.path.join(self.model_dir, 'model-%s.meta' % self.subdir)
        save_time_metagraph = 0  
        if not os.path.exists(metagraph_filename):
            print('Saving metagraph')
            start_time = time.time()
            self.saver.export_meta_graph(metagraph_filename)
            save_time_metagraph = time.time() - start_time
            print('Metagraph saved in %.2f seconds' % save_time_metagraph)
        self.summary = tf.Summary()
        #pylint: disable=maybe-no-member
        self.summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
        self.summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
        self.summary_writer.add_summary(self.summary, step)
    
      
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            