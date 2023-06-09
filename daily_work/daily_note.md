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
<font color='red'> 纠正： </font>卷积操作参数量计算方法为输出通道数C<sub>out</sub>×（输入通道数C<sub>in</sub>×长k<sub>W</sub>×高k<sub>H</sub>）+偏执，输出通道数即为卷积核数量，（k<sub>H</sub>k<sub>W</sub>C<sub>in</sub>）为卷积核形状。  
**PyTorch forward调用**  
out = net(input)调用\_\_call\_\_方法  
\_\_call\_\_方法中调用forward方法，由于每个网络定义时都重写了forward方法，所以都调用的是重写之后的forward方法
  


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
**trigger free memory callbacks函数**
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

### PyTorch模型的两种参数
模型中的参数保存在state\_dict()方法中，属于OrderDict类，是键值对的形式  
![](https://img-blog.csdnimg.cn/20210617213356840.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwMjA2Mzcx,size_16,color_FFFFFF,t_70)
**parameter**：反向传播时需要被optimizer更新，例如ConvNd,Linear,RNN，里面的权重参数会被自动认为Parameter，属于parameter类  
**buffer**：反向传播时不需要被optimizer更新，模型中不需要更新的参数注册会buffer保存在OrderDict中，便于模型转移设备时参数一起移动，属于tensor类

### PyTorch计算图
属性：边，表示操作或者操作的依赖；有输入边的点，表示一个操作；有输出边的点，表示一个变量  
分为静态图和动态图  
**静态图**：TensorFlow使用静态图。首先定义图的结构，然后为叶节点赋值（使用占位符placeholder）。根据叶结点的分配进行前向传播。  
**动态图**：PyTorch使用动态图。图结构在前向传播时逐步生成，无需占位符。每迭代一次都会构建一个新的计算图。PyTorch中，计算图用于反向传播计算参数梯度(autograd)。  
![](https://pic1.zhimg.com/v2-10fbe3014f69719d13d1f92aadd42ec8_b.webp)  
#### torch.FX
主要有三个组件：符号追踪器（symbolic tracer），中间表示（intermediate representation）， Python代码生成（Python code generation）。  
符号追踪器对模块的forward代码进行符号执行，它送入的是假的输入，叫做Proxies，代码中的所有operations都被记录下来。与TensorFlow构建静态图有点类似，Proxies类似placeholder。   
得到计算图的中间表示torch.fx.Graph，是静态图，记录了所有ops。  
一个Graph包括许多torch.fx.Node。Node是Gragh的基本单元，每个Node包含opcode，name，target，args，kwargs。Opcode记录操作类型，name记录操作名，target记录节点调用的对象，args和kwargs是节点的输入参数。  
最后根据Graph的语义自动生成相应的执行代码。  
 
通过symbolic\_trace函数，将Module映射到GraghModule。  
trace流程：  
1. 解析输入Module基本信息：输入的module赋值给self.root，对module的forward的函数签名（用来获取函数的参数列表，参数个数+参数类型+返回值）进行解析。  
2. 创建Gragh，初始化  
3. 为module的forward中除了self之外的每个输入参数创建Node和Proxy  
4. 使用动态属性替换（monkey patch），记录操作信息（），生成计算图节点，具体做法：  
&nbsp; 1. 暂时修改nn.Module类的静态方法\_\_getattr\_\_，拿到模型参数，parameter直接插入计算图中当叶子节点，buffer默认不插入（可调）。  
&nbsp; 2. 暂时修改nn.Module类的静态方法\_\_call\_\_，让原来执行网络运算的方法变为生成计算图的节点的方法。判断这个模块是不是叶子模块，是就插入节点；不是就递归进入模块内部继续追踪  
&nbsp; 3. 创建output节点。创建节点需要用到输入args，这部操作会递归调用4.2的call方法，直到找到叶子节点，从而递归生成整张计算图的节点。
综上，生成的图节点有6个类型：神经网络输入placeholder；模型参数get_attr；三种执行操作call\_function(自由函数，如+-\*/)；call\_module(模型操作，如Linear，ConvNd)；call\_method(torch函数，如ReLu)；神经网络输出output。

​

### PyTorch源码解析
C10：Caffe Tensor Library，最基础的张量库，包含PyTorch的核心抽象，包括张量和存储数据结构的实际实现  
ATen：A Tensor library，实现张量的操作

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

## GA解决工作流问题  
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

**Markdown插图**  图片名不能有中文，图片与笔记在同目录下，格式为\!\[](图片名)