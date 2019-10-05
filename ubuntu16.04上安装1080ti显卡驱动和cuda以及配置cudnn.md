# 安装显卡驱动

1.删除原有NVIDIA驱动
```
sudo apt-get remove --purge nvidia*
```
3.禁用nouveau
```
sudo gedit /etc/modprobe.d/blacklist.conf
```
在最后一行添加
```
blacklist nouneau
```
执行
```
sudo update-initramfs -u
```
重启
```
lsmod | grep nouveau # 没输出代表禁用生效,要在重启之后执行
```
4.查询自己的显卡型号
```
lshw -numeric -C display
```
截图如下
<div align="center">
<img src="./images/查看显卡信息.png">
<div>

5.下载适合自己显卡和系统的驱动：http://www.nvidia.cn/Download/index.aspx?lang=cn

6.给安装程序权限
```
sudo chmod 777 NVIDIA-Linux-x86_64-430.50.run
sudo ./NVIDIA-Linux-x86_64-430.50.run -no-opengl-files -no-x-checks -no-nouveau-check
```
7.驱动测试
```
nvidia-smi
```
截图如下
<div align="center">
<img src="./images/nvidia-smi.png">
<div>

# 安装cuda
