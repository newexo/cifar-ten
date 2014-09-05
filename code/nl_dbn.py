#import cPickle
import numpy as np
#from sklearn.externals import joblib
from nolearn.dbn import DBN
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from dataset import *

datasets = dataset.sharedTrain, dataset.sharedValid, dataset.sharedTest
train_set_x, train_set_y = dataset.sharedTrain
#valid_set_x, valid_set_y = dataset.sharedValid
test_set_x, test_set_y = dataset.sharedTest


n_feat = train_set_x.shape[1]
n_targets = train_set_y.max() + 1

net = DBN(
    [n_feat, n_feat / 3, n_targets],
    epochs=1000,
    learn_rates=0.01,
    learn_rate_decays=0.99,
    learn_rate_minimums=0.005,
    verbose=1,
    )
net.fit(train_set_x, train_set_y)


expected = test_set_y
predicted = net.predict(test_set_x)

print "Classification report for classifier %s:\n%s\n" % (
    net, classification_report(expected, predicted))
print "Confusion matrix:\n%s" % confusion_matrix(expected, predicted)
print "Accuracy score:",accuracy_score(expected,predicted)
