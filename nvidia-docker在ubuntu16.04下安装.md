# 安装docker
```
apt install docker 
```

# 安装nvidia-docker
```
# If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
sudo apt-get purge -y nvidia-docker

# Add the package repositories
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
$ sudo systemctl restart docker

# Test nvidia-smi with the latest official CUDA image
docker run --runtime=nvidia --rm nvidia/cuda:9.2-base nvidia-smi
```

# 针对CPU的测试(这里以yolov3模型的部署为例
windows下的脚本
```
docker run ^
--name yolov3 ^
-d ^
-p 8500:8500 ^
--mount type=bind,source=F:/project/tensorflow-yolov3-master/pb_model/,target=/models/yolov3 ^
--mount type=bind,source=F:/project/tensorflow-yolov3-master/models.config,target=/models/models.config ^
-t tensorflow/serving ^
--model_config_file=/models/models.config
```
ubuntu下的脚本
```
docker run \
--name yolov3 \
-d \
-p 8000:8500 \
--mount type=bind,source=$PWD/pb_model/,target=/models/yolov3 \
--mount type=bind,source=$PWD/models.config,target=/models/models.config \
-t tensorflow/serving \
--model_config_file=/models/models.config
```

# 针对GPU的测试(这里以yolov3模型的部署为例)
windows下的脚本
```
暂无，因为nvidia-docker不支持windows
```
ubuntu下的脚本
```
docker run \
--name yolov3 \
-d \
--runtime=nvidia \
-p 8000:8500 \
--mount type=bind,source=$PWD/pb_model/,target=/models/yolov3 \
--mount type=bind,source=$PWD/models.config,target=/models/models.config \
-e NVIDIA_VISIBLE_DEVICES=0 \
-t tensorflow/serving:1.11.0-gpu \
--model_config_file=/models/models.config
```

# 新增移除容器的脚本
windows下的脚本
```
docker stop yolov3 && docker rm yolov3
netstat -aon|findstr "8500"
```
ubuntu下的脚本
```
docker stop yolov3 && docker rm yolov3
netstat -tunlp |grep 8500
```
