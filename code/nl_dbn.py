import cPickle
import numpy as np
from sklearn.externals import joblib
from nolearn.dbn import DBN
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def load(name):
    with open(name, 'rb') as f:
        return cPickle.load(f)

dataset1 = load('/data/cifar10/cifar-10-batches-py/data_batch_1')
dataset2 = load('/data/cifar10/cifar-10-batches-py/data_batch_2')
dataset3 = load('/data/cifar10/cifar-10-batches-py/data_batch_3')
dataset4 = load('/data/cifar10/cifar-10-batches-py/data_batch_4')
dataset5 = load('/data/cifar10/cifar-10-batches-py/data_batch_5')
test_batch = load(/data/cifar10/cifar-10-batches-py/test_batch')

data_train = np.vstack([dataset1['data'], dataset2['data'], dataset3['data'],dataset4['data'],dataset5['data']])
labels_train = np.hstack([dataset1['labels'], dataset2['labels'],dataset3['labels'],dataset4['labels'],dataset5['labels']])

data_train = data_train.astype('float') / 255.
labels_train = labels_train
data_test = test_batch['data'].astype('float') / 255.
labels_test = np.array(test_batch['labels'])
n_feat = data_train.shape[1]
n_targets = labels_train.max() + 1

net = DBN(
    [n_feat, n_feat / 3, n_targets],
    epochs=1000,
    learn_rates=0.01,
    learn_rate_decays=0.99,
    learn_rate_minimums=0.005,
    verbose=1,
    )
net.fit(data_train, labels_train)


expected = labels_test
predicted = net.predict(data_test)

print "Classification report for classifier %s:\n%s\n" % (
    net, classification_report(expected, predicted))
print "Confusion matrix:\n%s" % confusion_matrix(expected, predicted)
