## benchmarks.py
### main
args接收命令行输入
实例化一个Benchmark对象b
遍历BENCHMARKS中的神经网络按照运行模式(eval,train)，batch\_size(1,32)

### Benchmark类
**load\_model方法**：加载已有神经网络，定义输入形状  
入参：模型名，执行模式，批量大小，设备名，profile=none，warm\_up\_iters=0预热器，profile\_iters=1，render\_model=False渲染模型，infer\_trace=False推断追踪  
(a,)定义一个一元组，必须加逗号
torch.concat(tuple,dim)按维度合并元组中的张量
首先判断模型名，给网络模型和模型输入初始化赋值  
判断是否在cpu上，不在就to(device)  
实例化一个TorchGraphImporter对象，并调用import\_via\_aotautograd方法，

**run\_simulation方法**：用graph和node\_rder预估一个显存峰值
入参是graph和node\_order
实例化了Simulator类并传入了graph,记录了开始时间start和结束时间stop,  
调用了类方法Simulate并传入node\_order,获得simulated\_peak\_mem\_usage, mem\_per\_timestep  
返回 预估显存峰值, 实例化Simulator（传入graph）时间

**run\_node\_ordering方法**：

## olla/torch/torch\_graph\_importer.py
**DeepTracer类**：提供了一个返回值为False的is\_leaf\_module方法  
### TorchGraphImporter类
变量后面加冒号是类型注释  
**import\_via\_fx方法**：
**import\_via\_aotautograd方法**：  
fn\_model\_wrapper方法：


## olla/simulator.py
**Simulate方法**：根据order遍历节点预估显存峰值  
入参是node\_ordering  
defaultdict(lambda: 0)定义一个默认 值为0的字典（值为int类型），lambda冒号后面是函数返回值。  
初始化一个默认字典用来记录张量引用次数  
初始化memory\_used记录当前显存使用  
遍历每个节点node，  
对节点的每个fanout，记录被引用次数，分配显存，显存占用增加  
比较显存占用与峰值占用，更新峰值占用，并将节点和显存占用作为元组加入到mem\_per\_timestep列表  
对节点的每个fanin，将其被引次数减1，被引次数不能小于0。如果被引次数归零，释放，显存占用减少  
返回显存峰值和mem\_per\_timestep列表