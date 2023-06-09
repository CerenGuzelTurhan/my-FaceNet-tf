import argparse
import logging
import os
import pickle
import sys
import time

import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from tensorflow.python.platform import gfile

import facenet

logger = logging.getLogger(__name__)

   
def main(input_dir, model_path = './models/facenets/20170512-110547/20170512-110547.pb', classifier_path = './models/facenet_embeddings/classifier.pkl',
         batch_size=128, num_threads=16, num_epochs=25, min_images_per_labels=40, split_ratio=0.4, is_train=False):
    """
    Loads images from :param input_dir, creates embeddings using a model defined at :param model_path, and trains
     a classifier outputted to :param output_path
     
    :param input_dir: Path to directory containing pre-processed images
    :param model_path: Path to protobuf graph file for facenet model
    :param classifier_path: Path to write pickled classifier
    :param batch_size: Batch size to create embeddings
    :param num_threads: Number of threads to utilize for queuing
    :param num_epochs: Number of epochs for each image
    :param min_images_per_labels: Minimum number of images per class
    :param split_ratio: Ratio to split train/test dataset
    :param is_train: bool denoting if training or evaluate
    """

    start_time = time.time()
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        train_set, test_set = _get_test_and_train_set(input_dir, min_num_images_per_label=min_images_per_labels,
                                                      split_ratio=split_ratio)
        if is_train:
            images, labels, class_names = _load_images_and_labels(train_set, image_size=160, batch_size=batch_size,
                                                                  num_threads=num_threads, num_epochs=num_epochs,
                                                                  random_flip=False, random_brightness=False,
                                                                  random_contrast=False)
            num_images = len(train_set)
        else:
            images, labels, class_names = _load_images_and_labels(test_set, image_size=160, batch_size=batch_size,
                                                                  num_threads=num_threads, num_epochs=1)
            num_images = len(test_set)

        _load_model(model_filepath=model_path)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embedding_layer = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        emb_array, label_array = _create_embeddings(embedding_layer, images, labels, images_placeholder,num_images, batch_size,
                                                    phase_train_placeholder, sess)

        coord.request_stop()
        coord.join(threads=threads)
        logger.info('Created {} embeddings'.format(len(emb_array)))

        classifier_filename = classifier_path

        if is_train:
            _train_and_save_classifier(emb_array, label_array, class_names, classifier_filename)
        else:
            acc = _evaluate_classifier(emb_array, label_array, classifier_filename)
            return acc, emb_array , label_array

        logger.info('Completed in {} seconds'.format(time.time() - start_time))


def _get_test_and_train_set(input_dir, min_num_images_per_label, split_ratio=0.7):
    """
    Load train and test dataset. Classes with < :param min_num_images_per_label will be filtered out.
    :param input_dir: 
    :param min_num_images_per_label: 
    :param split_ratio: 
    :return: 
    """
    dataset = facenet.get_dataset(input_dir)
    dataset = facenet.filter_dataset(dataset, min_images_per_label=min_num_images_per_label)
    train_set, test_set = facenet.split_dataset(dataset, split_ratio=split_ratio, mode = 'SPLIT_IMAGES' )

    return train_set, test_set


def _load_images_and_labels(dataset, image_size, batch_size, num_threads, num_epochs, random_flip=False,
                            random_brightness=False, random_contrast=False):
    class_names = [cls.name for cls in dataset]
    image_paths, labels = facenet.get_image_paths_and_labels(dataset)
    images, labels = facenet.read_data(image_paths, labels, image_size, batch_size, num_epochs, num_threads,
                                         shuffle=False, random_flip=random_flip, random_brightness=random_brightness,
                                         random_contrast=random_contrast)
    return images, labels, class_names


def _load_model(model_filepath , if_graph = True):
    """
    Load frozen protobuf graph
    :param model_filepath: Path to protobuf graph
    :type model_filepath: str
    """
    model_exp = os.path.normpath(os.path.join(os.getcwd(), model_filepath))
    
    #model_exp = os.path.expanduser(model_filepath)
    if if_graph:
        if os.path.isfile(model_exp):
            logging.info('Model filename: %s' % model_exp)
            with gfile.FastGFile(model_exp, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
        else:
            logger.error('Missing model file. Exiting')
            sys.exit(-1)
    else:
        facenet.load_model(model_exp)

def _create_embeddings(embedding_layer, images, labels, images_placeholder,num_images, batch_size, phase_train_placeholder, sess):
    """
    Uses model to generate embeddings from :param images.
    :param embedding_layer: 
    :param images: 
    :param labels: 
    :param images_placeholder: 
    :param phase_train_placeholder: 
    :param sess: 
    :return: (tuple): image embeddings and labels
    """
    emb_array = None
    label_array = None
    try:
        i = 0
        while True:
            batch_images, batch_labels = sess.run([images, labels])
            logger.info('Processing iteration {} batch of size: {}'.format(i, len(batch_labels)))
            emb = sess.run(embedding_layer,
                           feed_dict={images_placeholder: batch_images, phase_train_placeholder: False})

            emb_array = np.concatenate([emb_array, emb]) if emb_array is not None else emb
            label_array = np.concatenate([label_array, batch_labels]) if label_array is not None else batch_labels
            i += 1

    except tf.errors.OutOfRangeError:
        pass
# =============================================================================
#     try:
#         nrof_batches = int(math.ceil(1.0*num_images / batch_size))
#         print('Total number of batch: ', nrof_batches)
#         for i in range(nrof_batches):
#             batch_images, batch_labels = sess.run([images, labels])
#             logger.info('Processing iteration {} batch of size: {}'.format(i, len(batch_labels)))
#             emb = sess.run(embedding_layer,
#                            feed_dict={images_placeholder: batch_images, phase_train_placeholder: False})
# 
#             emb_array = np.concatenate([emb_array, emb]) if emb_array is not None else emb
#             label_array = np.concatenate([label_array, batch_labels]) if label_array is not None else batch_labels
# 
#     except tf.errors.OutOfRangeError:
#         pass
# 
# =============================================================================
    return emb_array, label_array


def _train_and_save_classifier(emb_array, label_array, class_names, classifier_filename_exp):
    logger.info('Training Classifier')
    model = SVC(kernel='linear', probability=True, verbose=False)
    model.fit(emb_array, label_array)
    
    classifier_filename_exp =  os.path.normpath(os.path.join(os.getcwd(),classifier_filename_exp))
    os.makedirs(os.path.dirname(classifier_filename_exp), exist_ok=True)
    with open(classifier_filename_exp, 'wb') as outfile:
        pickle.dump((model, class_names), outfile)
    logging.info('Saved classifier model to file "%s"' % classifier_filename_exp)


def _evaluate_classifier(emb_array, label_array, classifier_filename):
    logger.info('Evaluating classifier on {} images'.format(len(emb_array)))
    if not os.path.exists(classifier_filename):
        raise ValueError('Pickled classifier not found, have you trained first?')
    classifier_filename =  os.path.normpath(os.path.join(os.getcwd(),classifier_filename))
    with open(classifier_filename, 'rb') as f:
        model, class_names = pickle.load(f)

        predictions = model.predict_proba(emb_array, )
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        for i in range(len(best_class_indices)):
            print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

        accuracy = np.mean(np.equal(best_class_indices, label_array))
        acc_std =  np.std(np.equal(best_class_indices, label_array))
        print('Accuracy: %1.3f+-%1.3f' % (accuracy, acc_std))
        return accuracy

    
