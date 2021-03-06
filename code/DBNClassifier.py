import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import cifarDirectories
sys.path.append(cifarDirectories.DeepLearningTutorialsCode())
from DBN import DBN

from dataset import *
from hyperparameter import HyperparametersDBN

def test_DBN(dataset, hyper):
    datasets = dataset.sharedTrain, dataset.sharedValid, dataset.sharedTest

    train_set_x, train_set_y = dataset.sharedTrain
    valid_set_x, valid_set_y = dataset.sharedValid
    test_set_x, test_set_y = dataset.sharedTest

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / hyper.batchSize

    numpy_rng = numpy.random.RandomState(123)
    print '... building the model'

    dbn = DBN(numpy_rng=numpy_rng, n_ins=dataset.n_in,
              hidden_layers_sizes=hyper.nHidden,
              n_outs=dataset.n_out)

    print '... getting the pretraining functions'
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=hyper.batchSize,
                                                k=hyper.k)

    print '... pre-training the model'
    start_time = time.time()

    pretrainObjectives = []
    for i in xrange(dbn.n_layers):
        layerObjectives = []
        for epoch in xrange(hyper.pretrainingEpochs):
            t = (time.time() - start_time) / 60.0
            print 'Pretraining epoch %d, time %.2f' % (epoch, t)
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=hyper.pretrainingLearningRate))
            cost = abs(numpy.mean(c))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print cost
            layerObjectives.append((t, cost))
        pretrainObjectives.append(layerObjectives)

    end_time = time.time()
    print 'The pretraining code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.0)

    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = dbn.build_finetune_functions(
                datasets=datasets, batch_size=hyper.batchSize,
                learning_rate=hyper.learningRate)

    print '... finetuning the model'
    # early-stopping parameters
    patience = 4 * n_train_batches  # look as this many examples regardless
    validation_frequency = min(n_train_batches, patience / 2)

    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.time()

    done_looping = False
    epoch = 0

    finetuningObjectives = []
    while (epoch < hyper.numberEpochs) and (not done_looping):
        epoch = epoch + 1
        t = (time.time() - start_time) / 60.0
        print 'Finetuning epoch %d, time %.2f' % (epoch, t)
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %.2f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                if this_validation_loss < best_validation_loss:

                    if (this_validation_loss < best_validation_loss *
                        hyper.improvementThreshold):
                        patience = max(patience, iter * hyper.patienceIncrease)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %.2f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break
        finetuningObjectives.append((t, best_validation_loss, test_score))    
    end_time = time.time()
    print(('Optimization complete with best validation score of %.2f %%,'
           'with test performance %.2f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))

    return pretrainObjectives, finetuningObjectives


def interpretObjectives(objectives):
    pretrainObjectives, finetuningObjectives = objectives
    lastPretrainedLayer = pretrainObjectives[len(pretrainObjectives) - 1]
    pretrainingTime, cost = lastPretrainedLayer[len(lastPretrainedLayer) - 1]
    postpretrainerror = finetuningObjectives[0][1]
    print "Pretraining time %f and error %f" % (pretrainingTime, postpretrainerror)
    finetuningTime, best_validation_loss, test_score = finetuningObjectives[len(finetuningObjectives) - 1]
    print "Finetuning time %f, best validation loss %f and test score %f." % (finetuningTime, best_validation_loss, test_score)
    return pretrainingTime, postpretrainerror, finetuningTime, test_score

if __name__ == '__main__':
    hyper = HyperparametersDBN(learningRate=0.1,
                               numberEpochs=4,
                               pretrainingEpochs=1,
                               pretrainingLearningRate=0.1,
                               nHidden=[3000, 3000, 3000])
    logs = test_DBN(Mnist(), hyper)
    print logs
    print interpretObjectives(logs)
