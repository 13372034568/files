# 安装docker
```
• step 1: 安装必要的一些系统工具
sudo apt-get -y install apt-transport-https ca-certificates curl software-properties-common
• step 2: 安装GPG证书
curl -fsSL http://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo apt-key add -
• Step 3: 写入软件源信息
sudo add-apt-repository "deb [arch=amd64] http://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"
• Step 4: 更新并安装 docker-ce
sudo apt-get -y update
sudo apt-get -y install docker-ce
• 验证
sudo service docker status (检查Docker服务的状态) 
sudo docker run hello-world (测试Docker安装是否成功)
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
```
如果在此处出现错误（docker: Error response from daemon: Unknown runtime specified nvidia.），则尝试：
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd

再执行 docker run --runtime=nvidia --rm nvidia/cuda:9.2-base nvidia-smi
```

# 针对CPU的测试(这里以yolov3模型的部署为例)
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

# 移除容器的脚本
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

# models.config
```
model_config_list: {
    config: {
        name: "yolov3",
        base_path: "/models/yolov3",
        model_platform: "tensorflow",
        model_version_policy: {
          specific: {
              versions:1,
          }
        }
    },
}
```

# 调用grpc服务核心代码
```
import multiprocessing
import os
import random
import threading
import time
from grpc.beta import implementations
import tensorflow as tf
import json
from datetime import datetime
# from tensorflow_serving.apis import predict_pb2
# from tensorflow_serving.apis import prediction_service_pb2_grpc
import numpy as np
# from tensorboard._vendor.tensorflow_serving.apis import prediction_service_pb2_grpc, predict_pb2
from PIL import Image
import cv2
from tensorflow_serving.apis import prediction_service_pb2_grpc, predict_pb2

import core.utils as utils


class ModelResult(object):

    def __init__(self, host, port, model_name):
        self.host = host
        self.port = port
        self.model_name = model_name

    def model_reponse(self, data_string, original_image_size):
        channel = implementations.insecure_channel(self.host, int(self.port))  # 创建channel凭据
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel._channel)  # 利用.proto文件生成的类创建服务存根
        request = predict_pb2.PredictRequest()  # 请求类型
        request.model_spec.name = self.model_name  # 待评估模型的名称
        request.model_spec.signature_name = 'serving_default'  # 待评估模型的签名
        request.inputs['images'].CopyFrom(
            tf.contrib.util.make_tensor_proto(data_string, shape=[1, 416, 416, 3]))  # 输入数据格式转换
        result = stub.Predict(request, 10.0)
        sbbox = np.array(list(result.outputs['out1'].float_val))
        mbbox = np.array(list(result.outputs['out2'].float_val))
        lbbox = np.array(list(result.outputs['out3'].float_val))
        pred_bbox = np.concatenate([np.reshape(sbbox, (-1, 85)),
                                    np.reshape(mbbox, (-1, 85)),
                                    np.reshape(lbbox, (-1, 85))], axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, 416, 0.3)
        bboxes = utils.nms(bboxes, 0.15, method='nms')
        return bboxes


def detect_img(model_result, input_size, fp_src, fp_dst):
    original_image = cv2.imread(fp_src)
    original_image_draw = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]
    image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = np.array(image_data).astype(np.float32)
    bboxes = model_result.model_reponse(image_data, original_image_size)
    image = utils.draw_bbox(original_image_draw, bboxes)
    image = Image.fromarray(image)
    image.save(fp_dst)


def process_function(input_size, model_result, f_items, p_id, g_lst=None, share_var=None, share_lock=None):
    st = time.time()
    for e, f_item in enumerate(f_items):
        if e == 1:
            st = time.time()
        fp_src = f_item['src']
        fp_dst = f_item['dst']
        detect_img(model_result, input_size, fp_src, fp_dst)
        print('%s 处理完成图片 %s-->%s' % (p_id, fp_src, fp_src))
    ct = time.time() - st
    img_cnt = len(f_items) - 1
    at = ct / img_cnt
    if g_lst is not None:
        g_lst.append({'p_name': p_id, 'at': at, 'ct': ct, 'img_cnt': img_cnt})
    elif share_var is not None and share_lock is not None:
        # 获取锁
        share_lock.acquire()
        share_var.append({'p_name': p_id, 'at': at, 'ct': ct, 'img_cnt': img_cnt})
        # 释放锁
        share_lock.release()


# {'host': '220.178.172.160', 'port': 58000, 'model_name': 'yolov3'},
def g_options():
    grpc_options_cpu = [
                           {'host': '192.168.0.159', 'port': 8000, 'model_name': 'yolov3'},
                       ] * 6
    grpc_options_gpu = [
                           {'host': '192.168.0.159', 'port': 8001, 'model_name': 'yolov3'},
                       ] * 4
    grpc_options = []
    if grpc_options_cpu:
        grpc_options.extend(grpc_options_cpu)
    if grpc_options_gpu:
        grpc_options.extend(grpc_options_gpu)
    random.shuffle(grpc_options)
    grpc_cnt = len(grpc_options)
    return grpc_options, grpc_cnt


def g_model_results(grpc_options, grpc_cnt):
    dir_path_src = 'img_detect_all'
    dir_path_dst = 'img_out'
    fns = list(os.listdir(dir_path_src))
    f_items_lst = [[] for _ in range(grpc_cnt)]
    for e, fn in enumerate(fns):
        fp_src = os.path.join(dir_path_src, fn)
        fp_dst = os.path.join(dir_path_dst, fn)
        index = e % grpc_cnt
        f_items_lst[index].append({'src': fp_src, 'dst': fp_dst})
    model_results = [ModelResult(host=grpc_option['host'],
                                 port=grpc_option['port'],
                                 model_name=grpc_option['model_name']) for grpc_option in grpc_options]
    return model_results, f_items_lst


def g_result(share_var):
    share_var = list(share_var)
    share_var.sort(key=lambda x: x['p_name'], reverse=False)
    ct_max = 0
    img_cnt = 0
    for result in share_var:
        print('%s，平均时间为:%2f秒' % (result['p_name'], result['at']))
        ct_max = max(ct_max, result['ct'])
        img_cnt += result['img_cnt']
    g_at = ct_max / img_cnt
    print('总平均耗时为:%2f秒' % g_at)


def main_multi_thread():
    class MyThread(threading.Thread):
        def __init__(self, input_size, model_result, f_items, p_id, g_lst=None, share_var=None, share_lock=None):
            threading.Thread.__init__(self)
            self.input_size = input_size
            self.model_result = model_result
            self.f_items = f_items
            self.p_id = p_id
            self.g_lst = g_lst
            self.share_var = share_var
            self.share_lock = share_lock

        def run(self):
            process_function(self.input_size, self.model_result, self.f_items, self.p_id, self.g_lst, self.share_var,
                             self.share_lock)

    grpc_options, grpc_cnt = g_options()
    model_results, f_items_lst = g_model_results(grpc_options, grpc_cnt)
    input_size = 416
    if grpc_cnt == 1:
        share_var = []
        process_function(input_size, model_results[0], f_items_lst[0], p_id='单线程', g_lst=share_var)
    else:
        share_var = []
        share_lock = threading.Lock()
        threads = []
        for i in range(grpc_cnt):
            t = MyThread(input_size, model_results[i], f_items_lst[i], '线程%d' % i,
                         None, share_var, share_lock)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    g_result(share_var)


def main_multi_process():
    grpc_options, grpc_cnt = g_options()
    model_results, f_items_lst = g_model_results(grpc_options, grpc_cnt)
    input_size = 416
    if grpc_cnt == 1:
        share_var = []
        process_function(input_size, model_results[0], f_items_lst[0], p_id='单进程', g_lst=share_var)
    else:
        # 列表声明方式
        share_var = multiprocessing.Manager().list()
        # 声明一个共享锁
        share_lock = multiprocessing.Manager().Lock()
        pool = multiprocessing.Pool(grpc_cnt)
        for i in range(grpc_cnt):
            pool.apply_async(process_function, (input_size, model_results[i], f_items_lst[i], '进程%d' % i,
                                                None, share_var, share_lock))
        pool.close()
        pool.join()
    g_result(share_var)


if __name__ == '__main__':
    # main_multi_thread()
    main_multi_process()

```
