import os, tarfile
from PIL import Image
import numpy as np
import tqdm

dataset_directory="./dataset"
tar_path = "./data/lfw.tgz"
img_names=[]

def read_dataset(data_dir, is_tar = True, min_instance = 0): 
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        if is_tar:
            tar = tarfile.open(tar_path, 'r')
            for member in tar.getmembers():     
                tar.extract(member, data_dir)
                print("Dataset is ready")   
                
    img_paths = []
    labels =[]
    images = []
          
    for root, dirs, files in os.walk(data_dir):
        if files:
            for f in files:
                img_paths.append(os.path.join(root, f))
                img_names.append(f.split('_0')[0])  
                
    for files in img_paths:
        idx = img_paths.index(files)
        if img_names.count(img_names[idx]) >= min_instance:
            images.append(np.array(Image.open(files)))
            labels.append(img_names[idx])   
    print ("Images are loaded from ", data_dir, " folder")           
    return images, labels, img_paths     

def read_pairs(filename):
    pairs = []
    with open(filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)

def name2label(img_names):
    target_names = np.unique(img_names)
    labels = np.searchsorted(target_names, img_names)
    return labels

def label2image(labels, target_names):
    return target_names[labels]

def get_pairs(pairs, images, labels, target_names):
    pair_images = []
    pair_labels = []
    target_names = np.array(target_names)
    for pair in pairs:
        if len(pair) == 3:
            pair_class_ids = np.where(target_names == pair[0])[0]
            pair_images += (images[pair_class_ids[int(pair[1])-1]], images[pair_class_ids[int(pair[2])-1]])
            pair_labels += (labels[pair_class_ids[int(pair[1])-1]], labels[pair_class_ids[int(pair[2])-1]])
        if len(pair) == 4:
            pair_class1_ids = np.where(target_names == pair[0])[0]
            pair_class2_ids = np.where(target_names == pair[2])[0]
            pair_images += (images[pair_class1_ids[int(pair[1])-1]], images[pair_class2_ids[int(pair[3])-1]])
            pair_labels += (labels[pair_class1_ids[int(pair[1])-1]], labels[pair_class2_ids[int(pair[3])-1]])
    return pair_images, pair_labels

def get_pairpaths(dataset_dir, pairs):
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(dataset_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.jpg')
            path1 = os.path.join(dataset_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+'.jpg')
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(dataset_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.jpg')
            path1 = os.path.join(dataset_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+'.jpg')
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
    
    return path_list, issame_list

def save_dataset(images, path, img_names, img_paths):  
    if not os.path.exists(path):
        os.makedirs(path)
    for i in tqdm.tqdm(range(len(images))):    
        img = Image.fromarray(images[i])
        img.save(path + '/' +img_paths[i].split('./dataset/lfw' + '\\' + img_names[i] +'\\')[1])
    print ("Images are saved to ", path, " folder")
