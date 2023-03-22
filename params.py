dataset_dir= './dataset/lfw'
tar_path = './data/lfw.tgz'

is_preprocessed = False
is_extracted = False
is_train = True

detect_align_method = 'mtcnn' #'dlib' # 'haar' 
feature_extractor = None #"facenet_embeddings"  #None # "facenet_embeddings" ,"pca"
classifier =  'facenet' #'svm' # 'facenet' 'deepid3'

min_instance = 0
optimizer = 'ADAGRAD'
data_dir = '.../.../dataset/lfw_aligned_mtcnn' # SHOULD BE FULL PATH
lfw_dir = '..../.../dataset/lfw_aligned_mtcnn' # SHOULD BE FULL PATH
lfw_pairs = './data/pairs.txt'
lfw_test_pairs ='./data/pairsDevTest.txt'
lfw_train_pairs = './data/pairsDevTrain.txt'
facenet_embedding_dir = './models/facenet_embeddings/classifier.pkl'
epoch_size = 1000
batch_size = 100

pretrained_model='./models/facenets/20170512-110547'
max_nrof_epochs = 20
keep_probability = 0.8
learning_rate = 0.1
gpu_memory_fraction = 0.95
weight_decay = 2e-4
people_per_batch = 720
images_per_person = 5
image_size = 160
