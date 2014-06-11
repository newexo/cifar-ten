#!/bin/bash

function createDirectories()
{
    echo Creating data directories.

    mkdir data
    mkdir data/cifarKaggle
    mkdir data/cifarKaggle/test
    mkdir data/cifarKaggle/train
}

function downloadFiles()
{
    echo Downloading data.

    pushd .
    cd data

    if [ -e iris.data ]
    then
        echo Iris dataset already downloaded.
    else
        curl http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data > iris.data
    fi


    if [ -e mnist.pkl.gz ]
    then
        echo MNIST dataset already downloaded.
    else
        curl http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz > mnist.pkl.gz
    fi

    if [ -e cifar-10-python.tar.gz ]
    then
        echo CIFAR 10 dataset already downloaded.
    else
        curl http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz > cifar-10-python.tar.gz
    fi
    
    if [ -e cifar-100-python.tar.gz ]
    then
        echo CIFAR 100 dataset already downloaded.
    else
        curl http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz > cifar-100-python.tar.gz
    fi
    
    popd
}

function cloneRepos()
{
    pushd .
    cd demos
    
    if [ -e DeepLearningTutorials ]
    then
        echo DeepLearningTutorials repo already cloned.
    else
        git clone git@github.com:lisa-lab/DeepLearningTutorials.git
    fi
    
    if [ -e Vulpes ]
    then
        echo Vulpes repo already cloned.
    else
        git clone git@github.com:fsprojects/Vulpes.git
    fi
    
    if [ -e rbm-mnist ]
    then
        echo rbm-mnist repo already cloned.
    else
        git clone git@github.com:jdeng/rbm-mnist.git
    fi
    
    if [ -e pywt ]
    then
        echo pywt repo already cloned.
    else
        git clone git@github.com:nigma/pywt.git
    fi
    
    if [ -e cuda-convnet-read-only ]
    then
        echo cuda-convnet repo already cloned.
    else
        svn checkout http://cuda-convnet.googlecode.com/svn/trunk/ cuda-convnet-read-only
    fi
    
    popd
}

createDirectories
downloadFiles
cloneRepos

