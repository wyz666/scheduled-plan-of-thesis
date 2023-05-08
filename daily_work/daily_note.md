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

**PyTorch卷积原理**  
im2col算法：PyTorch，Caffe中卷积的实现都是基于一个im2col算法。将卷积运算转换为矩阵乘法运算。通过reshap将输入（C<sub>in</sub>, H<sub>in</sub>, W<sub>in</sub>）转为矩阵（H<sub>out</sub>W<sub>out</sub>, H<sub>k</sub>W<sub>k</sub>C<sub>in</sub>)，将C<sub>out</sub>个卷积核（H<sub>k</sub>, W<sub>k</sub>, C<sub>in</sub>）转为矩阵（H<sub>k</sub>W<sub>k</sub>C<sub>in</sub>, C<sub>out</sub>），输出一个（H<sub>out</sub>W<sub>out</sub>, C<sub>out</sub>）的矩阵，再reshape为（C<sub>out</sub>, H<sub>out</sub>, W<sub>out</sub>）  
<font color='red'> 纠正： </font>卷积操作参数量计算方法为输出通道数C<sub>out</sub>×（输入通道数C<sub>in</sub>×长k<sub>W</sub>×高k<sub>H</sub>）+偏执，输出通道数即为卷积核数量，（k<sub>H</sub>k<sub>W</sub>C<sub>in</sub>）为卷积核形状

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
**Block**  
Block是管理内存块的基本单位，由三元组(stream_id, size, ptr)定位，ptr决定内存地址，size决定大小，stram\_id决定为哪个CUDA流工作。  
所有连续的Block都被组织在一个双向链表里，以便将碎片合成整块。  
**BlockPool**  
内存池，用 std::set 存储 Block 的指针，按照 (cuda_stream_id -> block size -> addr) 的优先级从小到大排序，所有保存在 BlockPool 中的 Block 都是空闲的。  
DeviceCachingAllocator 中维护两种 BlockPool (large_blocks, small_blocks)，<1MB为小块，>1MB为大块。  
Block 在 Allocator 内有两种组织方式，一种是显式地组织在 BlockPool（红黑树）中，按照大小排列；另一种是具有连续地址的 Block 隐式地组织在一个双向链表里（通过结构体内的 prev, next 指针），可以以 O(1) 时间查找前后 Block 是否空闲，便于在释放当前 Block 时合并碎片。

**malloc函数**  
返回一个可用的Block
size：进程向缓存区请求分配的显存大小
alloc_size：缓存区实际提供的显存大小
get_allocation_size函数中明确：  
<1MB的size分配2MB；
<10MB的size分配20MB；
>=10MB的size分配(size+2MB-1)//2MB个2MB的显存。


分配存储空间时，从cache里找能满足的最小的block，然后对这个Block多余部分进行切分。  
不能满足时，用cudaMalloc向CUDA申请显存。 
cudaMalloc失败时，释放一个未被切分的最大的block。

### PyTorch实验心得
在学校服务器上远程同步本地代码，创建docker容器cu11(cuda11,cudnn8)  
已有imagenet数据集，格式为"目录/train/标签/标签\_图片号.JPEG"  
alexnet仓配置环境时loss.backword()报错，是因为torch版本问题,由1.11->1.2解决。  
运行时报错共享内存不足，原因时代码中使用了num\_workers来加速数据处理，而docker容器默认shm大小为64M，不够开这么多线程处理数据。更改容器的配置文件hostconfig.json解决。修改时需要关闭docker服务，systemctl stop/restart docker。  
**num\_workers能够显著提升训练速度**  
运行时显存占用维持在4255M，因为pytorch会维护一个缓存区，即使用不到也不会释放显存  
在shell中执行temp = torch.tensor([1.0]).cuda()可以测出CUDA context开销，我在我的conda环境alex下，pytorch版本1.2，开销为927M  

### GA解决工作流问题  
