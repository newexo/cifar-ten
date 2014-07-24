#!/bin/bash

export CIFAR10_HOME=$(pwd)

function experiment()
{
   	 echo Experiment $1. 
	 python randomDBN.py > "random$1.log"
}

pushd .
cd code

experiment 01
experiment 02
experiment 03
experiment 04

popd
