
#运行用户
user www-data;    
#启动进程,通常设置成和cpu的数量相等
worker_processes  1;
（3）     #全局错误日志及PID文件
error_log  /var/log/nginx/error.log;
pid        /var/run/nginx.pid;
（4）     #工作模式及连接数上限
events {
    use  epoll;            #epoll是多路复用IO(I/OMultiplexing)中的一种方式,但是仅用于linux2.6以上内核,可以大大提高nginx的性能
    worker_connections  1024;#单个后台worker process进程的最大并发链接数
    # multi_accept on; 
}
（5）     #设定http服务器，利用它的反向代理功能提供负载均衡支持
http {
     #设定mime类型,类型由mime.type文件定义
    include      /etc/nginx/mime.types;
    default_type  application/octet-stream;
    #设定日志格式
    access_log    /var/log/nginx/access.log;
（6）         #sendfile 指令指定 nginx 是否调用 sendfile 函数（zero copy 方式）来输出文件，对于普通应用，
    #必须设为 on,如果用来进行下载等应用磁盘IO重负载应用，可设置为 off，以平衡磁盘与网络I/O处理速度，降低系统的uptime.
    sendfile        on;
    #tcp_nopush     on;
（7）         #连接超时时间
    #keepalive_timeout  0;
    keepalive_timeout  65;
    tcp_nodelay        on;

    #开启gzip压缩
    gzip  on;
    gzip_disable "MSIE [1-6]\.(?!.*SV1)";
（8）         #设定请求缓冲
    client_header_buffer_size    1k;
    large_client_header_buffers  4 4k;
（9）         include /etc/nginx/conf.d/*.conf;
    include /etc/nginx/sites-enabled/*;
（10）      #设定负载均衡的服务器列表
     upstream mysvr {
    #weigth参数表示权值，权值越高被分配到的几率越大
    #本机上的Squid开启3128端口
    server 192.168.8.1:3128 weight=5;
    server 192.168.8.2:80  weight=1;
    server 192.168.8.3:80  weight=6;
    }
（11）  
   server {
    #侦听80端口
       listen       80;
        #定义使用www.xx.com访问
        server_name  www.xx.com;
（12）          #设定本虚拟主机的访问日志
        access_log logs/www.xx.com.access.log  main;
（13）      #默认请求
    location / {
          root  /root;      #定义服务器的默认网站根目录位置
          index index.phpindex.html index.htm;  #定义首页索引文件的名称
#请求转向定义的服务器列表 
proxy_pass  http://mysvr ;
（14）           fastcgi_pass  www.xx.com;
         fastcgi_param SCRIPT_FILENAME  $document_root/$fastcgi_script_name; 
          include/etc/nginx/fastcgi_params;
        }
（15）      # 定义错误提示页面
    error_page   500 502 503 504 /50x.html;  
        location = /50x.html {
        root   /root;
    }
（16）      #静态文件，nginx自己处理
    location ~ ^/(images|javascript|js|css|flash|media|static)/{
        root /var/www/virtual/htdocs;
        #过期30天，静态文件不怎么更新，过期可以设大一点，如果频繁更新，则可以设置得小一点。
        expires 30d;
    }
    #PHP 脚本请求全部转发到 FastCGI处理. 使用FastCGI默认配置.
    location ~ \.php$ {
        root /root;
        fastcgi_pass 127.0.0.1:9000;
        fastcgi_index index.php;
        fastcgi_param SCRIPT_FILENAME/home/www/www$fastcgi_script_name;
        include fastcgi_params;
    }
    #设定查看Nginx状态的地址
    location /NginxStatus {
       stub_status           on;
       access_log             on;
       auth_basic             "NginxStatus";
        auth_basic_user_file conf/htpasswd;
    }
    #禁止访问 .htxxx 文件
    location ~ /\.ht {
        deny all;
    }

     }
}
