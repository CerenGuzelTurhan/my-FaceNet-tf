from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np

def recognize(dataset):
    if len(dataset) == 2:
        [X, Y] = dataset
        target_names = np.unique(Y)
        labels = np.searchsorted(target_names, Y)
        images = np.reshape(X, (X.shape[0], -1))
        trainX, testX, trainY, testY = train_test_split(images, labels, test_size=0.25, random_state=42)
    if len(dataset) == 4:    
        [trainX, testX, trainY, testY] = dataset
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(trainX, trainY)
    predY = clf.predict(testX)
    acc_report = classification_report(testY, predY)
    correct = (predY == testY)
    acc = correct.sum() / correct.size
    return acc, acc_report