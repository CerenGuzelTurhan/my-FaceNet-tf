import numpy as np
import dataio as d

import detectandalign.dlib_detector as dlib_detector
import detectandalign.haar_detector as haar_detector
import detectandalign.haar_aligner as haar_aligner
import detectandalign.dlib_aligner as dlib_aligner
import detectandalign.aligner_mtcnn as mtcnn_aligner
import extract.PCA_feature_extractor as PCA
import classify.SVM_classifier as SVM
import train_tripletloss
import train_deepid3
import validate_on_lfw
#import extract_embeddings
import create_embeddings as FacenetEmbeddingClassifier

import params as p

# Function renaming for simplicity
if p.detect_align_method == "haar":
    faceDet = haar_detector 
    aligner = haar_aligner 
elif p.detect_align_method == "dlib":
    faceDet = dlib_detector 
    aligner = dlib_aligner
else:
    aligner = mtcnn_aligner  
    
if p.feature_extractor == 'pca':
    featureExtractor = PCA
if p.feature_extractor == 'facenet_embeddings':
    featureExtractor = FacenetEmbeddingClassifier
    
if p.classifier =='svm':
    facialRecognizer = SVM
elif p.classifier == 'facenet':
     facialRecognizer = train_tripletloss
else:
    model1 = train_deepid3.main(model_name = 'models.deepid3_net1',max_nrof_epochs = p.max_nrof_epochs,  data_dir = p.lfw_dir, lfw_dir = p.lfw_dir, gpu_memory_fraction = p.gpu_memory_fraction, lfw_pairs = p.lfw_pairs,lfw_train_pairs = p.lfw_train_pairs)
    model2 = train_deepid3.main(model_name = 'models.deepid3_net2',max_nrof_epochs = p.max_nrof_epochs,  data_dir = p.lfw_dir, lfw_dir = p.lfw_dir, gpu_memory_fraction = p.gpu_memory_fraction, lfw_pairs = p.lfw_pairs,lfw_train_pairs = p.lfw_train_pairs)
    extracted_dataset1 = featureExtractor.extractor(data_dir = p.data_dir, model = model1)
    extracted_dataset2 = featureExtractor.extractor(data_dir = p.data_dir, model = model2)
    extracted_dataset = [np.concatenate(extracted_dataset1[0], extracted_dataset2[0]), np.concatenate(extracted_dataset1[1], extracted_dataset2[1]), extracted_dataset1[2], extracted_dataset1[3]]
    facialRecognizer = SVM
    
# Face recognition pipeline    
    
if p.is_preprocessed:       
    faceDet.detect()
    aligner.align()
  
if p.feature_extractor == 'pca' and  p.classifier =='svm':
    extracted_dataset = featureExtractor.extractor(data_dir = p.data_dir)       
    result, _ = facialRecognizer.recognize(extracted_dataset)
elif p.feature_extractor == 'facenet_embeddings' and p.classifier =='svm':
    if p.is_train:
        featureExtractor.main(input_dir =p.lfw_dir, classifier_path = p.facenet_embedding_dir,is_train=True)
    else:
        result, _, _ =  featureExtractor.main(input_dir =p.lfw_dir, classifier_path = p.facenet_embedding_dir)

if p.classifier =='svm':
    if not p.is_extracted:
        images, labels, _ = d.read_dataset(p.data_dir)
        result, _ = facialRecognizer.recognize([images, labels])
      
else:
    if p.classifier =="facenet":
        if p.is_train:
            facialRecognizer.main(optimizer = p.optimizer, data_dir = p.data_dir, lfw_dir = p.lfw_dir, max_nrof_epochs = p.max_nrof_epochs, keep_probability = p.keep_probability, learning_rate=p.learning_rate, 
                                      gpu_memory_fraction = p.gpu_memory_fraction, weight_decay = p.weight_decay, people_per_batch = p.people_per_batch/5, batch_size = p.batch_size, epoch_size = p.epoch_size/2, images_per_person = p.images_per_person,
                                      lfw_pairs = p.lfw_pairs,lfw_train_pairs = None)
    else:
        if p.is_train:
            result, _ = facialRecognizer.recognize(extracted_dataset)

    if not p.is_train:
         facialRecognizer.recognize = validate_on_lfw.main
         result = facialRecognizer.recognize(lfw_dir =  p.lfw_dir, model = p.pretrained_model, lfw_batch_size = p.lfw_batch_size, image_size = p.image_size, lfw_pairs= p.lfw_pairs)

