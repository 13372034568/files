docker run ^
--name yolov3 ^
-d ^
-p 8500:8500 ^
--mount type=bind,source=F:/project/tensorflow-yolov3-master/pb_model/,target=/models/yolov3 ^
--mount type=bind,source=F:/project/tensorflow-yolov3-master/models.config,target=/models/models.config ^
-t tensorflow/serving ^
--model_config_file=/models/models.config
