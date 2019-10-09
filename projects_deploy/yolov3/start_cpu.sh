#!/bin/bash

sudo docker run \
--name yolov3_cpu \
-d \
-p 8000:8500 \
--mount type=bind,source=$PWD/pb_model/,target=/models/yolov3 \
--mount type=bind,source=$PWD/models_cpu.config,target=/models/models.config \
-t tensorflow/serving \
--model_config_file=/models/models.config
