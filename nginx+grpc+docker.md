nginx在1.13版本之后可以支持grpc的负载均衡

所以，如果不用docker，则需要手动下载nginx1.14.1：http://nginx.org/download/nginx-1.14.1.tar.gz
# 编译和安装nginx
参考链接：
ubuntu16.04源码编译安装nginx1.14.2：https://www.cnblogs.com/xwgcxk/p/10973645.html

源码编译更新nginx到最新版本，并开始nginx支持http2协议模块：https://yq.aliyun.com/articles/117130?t=t1

解压后，进入nginx目录,执行
```
sudo apt-get update   
sudo apt-get install libpcre3 libpcre3-dev 
apt-get install zlib1g-dev
```
安装gcc
```
sudo apt-get  install  build-essential
```
安装好之后继续执行
```
./configure --with-http_v2_module
这里之所以加上后缀，是因为grpc在nginx中写作http2，如果不加，会报错
```
执行
```
make
make install
```
切到路径：/usr/local/nginx/sbin

配置软链接
```
sudo ln -s /usr/local/nginx/sbin/nginx /usr/bin/nginx
```
测试nginx配置文件是否正确
```
nginx -t -c /usr/local/nginx/conf/nginx.conf
```
启动nginx
```
nginx -c /usr/local/nginx/conf/nginx.conf
```
关闭nginx
```
nginx -s stop
```
# 从docker安装nginx
参考链接：https://www.runoob.com/docker/docker-install-nginx.html
```
docker pull nginx， 默认安装latest
```
以下命令使用 NGINX 默认的配置来启动一个 Nginx 容器实例
```
docker run --name runoob-nginx-test -p 8081:80 -d nginx
```
创建目录 nginx
```
mkdir -p ~/nginx/conf
```
拷贝容器内 Nginx 默认配置文件到本地当前目录下的 conf 目录, 容器 ID 可以查看 docker ps 命令输入中的第一列
```
docker cp 6dd4380ba708:/etc/nginx/nginx.conf ~/nginx/conf
```
部署命令
```
docker run -d -p 8082:80 --name nginx-grpc \
-v ~/nginx/conf/nginx.conf:/etc/nginx/nginx.conf \
nginx
```
# 修改nginx.conf,设置grpc负载
参考链接：
解决Nginx错误信息：client intended to send too large body：http://baijiahao.baidu.com/s?id=1600418962381598055&wfr=spider&for=pc

【部署问题】解决Nginx: [error] open() ＂/usr/local/Nginx/logs/Nginx.pid" failed：https://www.cnblogs.com/iloverain/p/9428630.html

nginx grpc streaming负载均衡的排坑和思考：http://xiaorui.cc/2019/07/27/nginx-grpc-streaming%e8%b4%9f%e8%bd%bd%e5%9d%87%e8%a1%a1%e7%9a%84%e6%8e%92%e5%9d%91%e5%92%8c%e6%80%9d%e8%80%83/
```
在http下修改两个地方
1. 因为要传图片，所以要增加client_max_body_size的设置
2. 尾部要增加新的server
```
```
http {
    ...
    keepalive_timeout  65;

	client_max_body_size 20M;

   ...
	server {
	    listen       6666 http2;
	    server_name  localhost;

	    location / {
		      grpc_pass grpc://grpcservers;
	    }
	}

	upstream grpcservers {
	    server 127.0.0.1:8000;
	    server 127.0.0.1:8001;
	    server 192.168.1.75:8000;
	    keepalive 2000;
	}

}
```

修改完后重新启动容器或者nginx即可，对外端口即为6666

# 测试端口
```
curl http://localhost:6666
```

