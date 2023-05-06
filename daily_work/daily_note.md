pycharm远端连接docker内虚拟环境
工具：pycharm专业版


新建一个dockers容器
docker run --gpus all -p 7600:22 --name toch1.8-cu11 
-v /home/yanze/test:/home/yanze/test 
-it meadml/cuda11.1-cudnn8-devel-ubuntu18.04-python3.8:latest /bin/bash
