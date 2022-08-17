#!/bin/bash

for models in resnet20 resnet32 resnet56
do
	echo "python resnet_trainer.py  --arch=$models --dataset=cifar100"
	python resnet_trainer.py  --arch=$models --dataset=cifar100 --runType=nc
done
