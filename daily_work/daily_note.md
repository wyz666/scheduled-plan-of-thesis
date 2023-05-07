## pycharm连接远程服务器
### 工具：pycharm专业版  
tools->Deployment->  
Configuration 配置ssh远程连接信息：服务器IP地址，密钥，mapping关联文件夹  
Options 设置同步细节


## Linux新建容器
### 工具：docker
常用命令：iamges,pull,rmi,ps,run,start,attach,exec,stop,rm   
新建一个dockers容器,-p指定端口连接，-v关联宿主机文件夹  
docker run --gpus all -p 7600:22 --name toch1.8-cu11 
-v /home/yanze/test:/home/yanze/test 
-it meadml/cuda11.1-cudnn8-devel-ubuntu18.04-python3.8:latest /bin/bash

## 跑通alexnet
跑通了但是精确度每次都一样

## PyTorch显存分配原理研究

### 背景知识
**神经网络训练流程**  
前向传播：Y=act(WX+b),求出Loss(output)  
**反向传播理解**：对W，b求导准备梯度更新，更新W需要求Loss对W的偏导，例如X<sub>n</sub>到Y<sub>m</sub>的参数W<sub>nm</sub>，L对W<sub>nm</sub>的偏导就是L对Y<sub>m</sub>的偏导（即反向传播每个神经元的值）乘上Y<sub>m</sub>对W<sub>nm</sub>的偏导（即前向传播的X<sub>n</sub>）  
梯度更新：根据学习率和导数更新权重和偏置。这里会使用优化器（SGD，Adam），可以加速训练，有额外的显存开销。  

**GPU一些概念**  
Stream（流）：一系列顺序执行的命令，流之间无序、并发  
Streaming Processor(SP)：基本处理单元，硬件概念，一个SP对应一个thread
Streaming Multiprocessor(SM)：由多个SP组成，加一些存储资源，共享内存，寄存器等。一个SM中的所有SP先分成warp，再共享内存和指令单元  
Warp：SP在SM中的物理分组，Tegra（图睿，nvidia产品线）一个warp有32个SP（thread），32个SP同时执行相同指令，没那么多工作就静默。warp是最小的硬件执行单位，所以一般thread设为32的倍数，这也是batch_size设为32倍数的原因
Grid、Block和Thread：软件概念。一个grid分为多个block，一个block分为多个thread。

**CUDA流**
一系列异步CUDA操作，例如分配设备内存cudaMalloc，主机与设备之间传输数据，主->设cudaMemcpy，主<-设Memcpy，核函数启动。  
流能封装这些异步操作，并保持操作顺序，有了流，就能查询排队状态了。


### 显存分析方法
torch.cuda.memory_allocated()：当前进程中torch.Tensor所占用的GPU显存  
torch.cuda.max_memory_allocated()：到调用该函数为止最大的显存占用字节数  
torch.cuda.memory_reserved()：查看当前进程所分配的显存缓冲区是多少  
### PyTorch显存分配机制
**多级分配机制**  
PyTorch分配显存时，会先向CUDA（GPU）申请MB为单位的空间放入到Cached Memory中，然后再为进程分配Memory。GPU(CUDA)->cached mem->allocated mem
**基本单位：Block  **

分配存储空间时，从cache里找能满足的最小的block，然后对这个block多余部分进行切分。  
不能满足时，用cudaMalloc向CUDA申请显存。 
cudaMalloc失败时，释放一个未被切分的最大的block。