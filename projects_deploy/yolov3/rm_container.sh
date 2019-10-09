#!/bin/bash

sudo docker stop yolov3 && docker rm yolov3
sudo docker stop yolov3_cpu && docker rm yolov3_cpu
sudo docker stop yolov3_gpu && docker rm yolov3_gpu
sudo netstat -tunlp |grep 8500

