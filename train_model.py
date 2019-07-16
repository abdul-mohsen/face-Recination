# USAGE
# python train_model.py --embeddings output/embeddings.pickle \
#	--recognizer output/recognizer.pickle --le output/le.pickle

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# from sklearn import cross_validation
from sklearn.svm import SVC
import argparse
import pickle
import os
from time import time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=False, default='output/embeddings.pickle',
	help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=False, default= 'output/recognizer.pickle',
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=False, default= 'output/le.pickle',
	help="path to output label encoder")
args = vars(ap.parse_args())

# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

X_train, X_test, y_train, y_test = train_test_split(
    data["embeddings"], labels,stratify=labels, test_size=0.25)

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e1,5e2,1e3, 5e3, 1e4, 5e4],
              'gamma': [0.00005,0.0001, 0.0005, 0.001, 0.005],
              'tol':[1e-4,1e-5,1e-6] }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced',probability=True),
                   param_grid, cv=5, iid=False)
clf = clf.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test)
print("done in %0.3fs" % (time() - t0))
print(classification_report(y_test, y_pred, target_names=os.listdir("dataset")))
print(confusion_matrix(y_test, y_pred, labels=range(len(os.listdir("dataset")))))

# skf = cross_validation.StratifiedKFold(labels, n_folds=10)
# X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.25)
# for train_index, test_index in skf:
# 	print("TRAIN:", train_index, "TEST:", test_index)
# 	X_train, X_test = X[train_index], X[test_index]
# 	y_train, y_test = y[train_index], y[test_index]


# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
# print("[INFO] training model...")
# recognizer = SVC(C=1.5, kernel="rbf",gamma='auto', probability=True,tol=1e-4)
# recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(clf))
f.close()

# write the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()