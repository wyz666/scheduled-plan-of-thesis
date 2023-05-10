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
Dropout：训练时，随即丢弃一些神经元  
Batch normalization（BN）：训练时，数据标准化。把数据均匀分布到激活函数敏感区域

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
torch.cuda.memory\_allocated()：当前进程中torch.Tensor所占用的GPU显存  
torch.cuda.max\_memory\_allocated()：到调用该函数为止最大的显存占用字节数  
torch.cuda.memory\_reserved()：查看当前进程所分配的显存缓冲区是多少  
### PyTorch显存分配机制
**多级分配机制**  
PyTorch分配显存时，会先向CUDA（GPU）申请MB为单位的空间放入到Cached Memory中，然后再为进程分配Memory。GPU(CUDA)->cached mem->allocated mem
**Block**  
Block是管理内存块的基本单位，由三元组(stream\_id, size, ptr)定位，ptr决定内存地址，size决定大小，stram\_id决定为哪个CUDA流工作。  
所有连续的Block都被组织在一个双向链表里，以便将碎片合成整块。  
**BlockPool**  
内存池，用 std::set 存储 Block 的指针，按照 (cuda\_stream\_id -> block size -> addr) 的优先级从小到大排序，所有保存在 BlockPool 中的 Block 都是空闲的。  
DeviceCachingAllocator 中维护两种 BlockPool (large\_blocks, small\_blocks)，<1MB为小块，>1MB为大块。  
Block 在 Allocator 内有两种组织方式，一种是显式地组织在 BlockPool（红黑树）中，按照大小排列；另一种是具有连续地址的 Block 隐式地组织在一个双向链表里（通过结构体内的 prev, next 指针），可以以 O(1) 时间查找前后 Block 是否空闲，便于在释放当前 Block 时合并碎片。  
**get free block函数**  
找能满足size的且大小相差不是太大的block(相差不超过20MB)  
**trigger free memory callbacks函数  **
主动调用collect函数，将缓存区reference为0的blocks回收（使用过的后续不会调用的，不回收就闲置在缓存区也不会被分配），不让这些blocks闲置，再执行get free block  
**alloc block函数**  
上两步都执行完了实在无可分配block了，调用cudaMalloc向显存要新的block。  
返回一个可用的Block  
size：进程向缓存区请求分配的显存大小  
alloc_size：缓存区向显存请求的，显存实际返回的block大小  
get_allocation_size函数中明确：  
<1MB的size分配2MB；  
<10MB的size分配20MB；  
\>=10MB的size分配(size+2MB-1)//2MB个2MB的显存。  
**release available cached blocks函数**  
释放一些较大的blocks，再进行cudaMalloc  
**release cached blocks函数**  
释放所有blocks，再cudaMalloc。与上一步不同在于把大小在阈值以下的blocks也都释放掉  

alloc block返回的block大小会大于实际申请大小，所以在分配后会进行拆分。且新的block无法保证与之前的block地址连续，无法写在双链表中。   
每当一个block被释放时，会判断前后是否有空闲块，有就合并减少碎片。  
结论：碎片问题突出

### Tensor
PyTorch张量默认存储到CPU上，用cuda方法转移到指定GPU上。  
pytorch中，一个tensor分为信息区Tensor和存储区Storage。信息区保存形状，步长，数据类型等信息。Storage将数据保存成连续数组，存在存储区。


### PyTorch计算图


### PyTorch实验心得
在学校服务器上远程同步本地代码，创建docker容器cu11(cuda11,cudnn8)  
已有imagenet数据集，格式为"目录/train/标签/标签\_图片号.JPEG"  
alexnet仓配置环境时loss.backword()报错，是因为torch版本问题,由1.11->1.2解决。  
运行时报错共享内存不足，原因时代码中使用了num\_workers来加速数据处理，而docker容器默认shm大小为64M，不够开这么多线程处理数据。更改容器的配置文件hostconfig.json解决。修改时需要关闭docker服务，systemctl stop/restart docker。  
**num\_workers能够显著提升训练速度**  
运行时显存占用维持在4255M，因为pytorch会维护一个缓存区，即使用不到也不会释放显存  
**PyTorch context：**  
CUDA context，就是在第一次执行CUDA操作，也就是使用GPU的时候所需要创建的维护设备间工作的一些相关信息。即CUDA running时固有配件必须要占掉的显存。  
在shell中执行temp = torch.tensor([1.0]).cuda()可以测出CUDA context开销，在我的conda环境alex下，pytorch版本1.2，开销为927M  

### GA解决工作流问题  
Grid Workflow Scheduling Based on Improved Genetic Algorithm  
改进遗传算法，网格工作流调度，有向无环图a directed acyclic graph (DAG)  
问题描述：将工作流描述为一系列子任务。  
子任务之间的依赖关系和优先级关系生成一个有向无环图G = (V, E, W)。节点V表示工作流所有子任务，有向边E表示人物之间依赖关系核优先级关系，有向边权值W表示节点V<sub>i</sub>所需代价。关键路径为运行时间最长的路径。  
将n个任务T<sub>i</sub>（i=1,2,...,n）分配给m个网格资源R<sub>j</sub>（i=1,2,...,n），C<sub>i</sub>表示T<sub>i</sub>的完成时间，目标是最小化∑C<sub>i</sub>，约束条件是依赖关系以及资源是否可用  
**GA**  
每条染色体代表一个调度方案。  
改进算法采用自然数编码，增加了二次优先杂交和二次优先突变过程，并在每一代繁育后保留最佳个体，有效保证了较强的全局搜索和局部搜索性能。  
编码：每个task有一个编号（1~n），每个task有一个二元组（s<sub>i</sub>,p<sub>i</sub>）,s表示分配的资源，p表示task的优先级
fitness：运行时间的倒数，未给详细推导公式  
选择使用轮盘赌，保留最优个体  
交叉使用单点交叉，但这篇文章使用二次交叉，对孩子进行评估，劣于平均值会与本代最优个体再进行一次交叉
变异使用单点变异，也用了评估+二次变异的操作  
**实验**  
他是使用工具做的仿真，跟标准遗传算法做对比。  
**思考**  
该文主要是优化了遗传算法，对工作流的研究不是很深入  
建模方式还需要研究一下pytorch计算流图  
我对神经网络建模后是否可以对网络进行一个仿真训练得到显存占用？  
我可以借鉴一些改良的遗传算法