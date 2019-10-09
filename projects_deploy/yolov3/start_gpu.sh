#!/bin/bash

sudo docker run \
--name yolov3_gpu \
-d \
--runtime=nvidia \
-p 8001:8500 \
--mount type=bind,source=$PWD/pb_model/,target=/models/yolov3 \
--mount type=bind,source=$PWD/models_gpu.config,target=/models/models.config \
-e NVIDIA_VISIBLE_DEVICES=0 \
-t tensorflow/serving:1.11.0-gpu \
--model_config_file=/models/models.config
