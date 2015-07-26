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

git submodule init
git submodule update

createDirectories
downloadFiles

