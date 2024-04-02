#!/bin/bash

sudo docker build -t deploy-ot . 

sudo docker tag deploy-ot invocation02/memetic-2024:buildv1

sudo docker push invocation02/memetic-2024:buildv1