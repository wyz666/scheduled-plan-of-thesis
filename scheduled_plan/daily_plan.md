
## 5.5任务
#### 新建一个github仓  
里面是两个文件夹  
.git  
- daily work  
- plan  
	-- 八周计划.md  
	-- 明日计划.md  
<font color='LimeGreen'> 完成 </font>  

#### 运行lenet alexnet 找几个地方打显存输出，同步打出当时的时间，做初步研究  
<font color='LimeGreen'> 5.5晚注: </font>lenet环境配置完成， 遗留问题： 服务器cuda没配，部署在学校GPU上<font color='LimeGreen'> 5.6已解决 </font>   
<font color='LimeGreen'> 5.6晚注: </font>lenet远程同步完成，调用GPU完成，打印各层显存分配输出到log文件夹中。  <font color='red'> 遗留问题： </font>最大显存占用为什么过大（384）。

alexnet运行  
遗留问题： 每次运行的精度输出相同<font color='LimeGreen'> 5.7已解决 </font>  
<font color='LimeGreen'> 5.7晚注: </font>目前找的库没有实现train功能，代码使用网上下载的预训练模型参数上传至github，弃用，需要另外找，同时用git管理

#### 打印论文 
<font color='LimeGreen'> 完成 </font>


#### 找一篇非常好的博客
讲解pytorch显存的，分享给文恺  
<font color='LimeGreen'> 5.5晚注: </font>需要再找一篇交叉比较<font color='LimeGreen'> 5.7已解决 </font>  


#### 快速过一遍老师的论文
进行总结  
<font color='LimeGreen'> 完成 </font>

#### 找一篇关于多目标的博客或者知乎
分享  
<font color='red'> 未完成 </font>

【有任何不会的问题，第一时间百度】

## 5.6任务  

#### cv领域主要研究中间结果的显存占用，nlp相对要多研究一个模型参数（需要调研，搜索引擎）  
<font color='red'> 未完成 </font>

#### 在生成中间结果时，pytorch对该张量进行哪些操作？内存分配，内存释放（看代码，看实现方法）  
<font color='red'> 未完成 </font>

#### 在模型训练前，确认cuda的上下文是怎么出来的？（看代码）  
<font color='LimeGreen'> 完成 </font>注：CUDA context，CUDA运行时固有配件必须要占用的显存，无需研究

pytorch视频讲解

## 5.7任务

#### 找一个markdown编译器软件，
<font color='LimeGreen'> 完成 </font>

#### 学习notion入门 
<font color='yellow'> 弃用 </font>

#### 补进度  

#### 搞清楚显存峰值过大的原因 
<font color='red'> 未完成 </font> 

#### log同步到本地 
<font color='LimeGreen'> 完成 </font>  
#### sudo权限 
<font color='LimeGreen'> 完成 </font>

## 5.8任务

#### alexnet找一个新仓，git管理，远程同步
<font color='LimeGreen'> 完成 </font>

#### 卷积层反向传播，https://blog.csdn.net/qq_43409114/article/details/105426806
<font color='LimeGreen'> 完成 </font>

#### 继续看pytorch显存调度
<font color='LimeGreen'> 完成 </font>

#### 调研遗传算法解决工作流问题，整理出汇报文档
<font color='LimeGreen'> 完成 </font>

## 5.9任务

####调研遗传算法解决工作流问题，整理出汇报文档
<font color='LimeGreen'> 完成 </font>
#### 继续看pytorch显存调度
<font color='LimeGreen'> 完成 </font>
#### 汇报进度，得到反馈
<font color='LimeGreen'> 完成 </font>   
调研pytorch生成神经网络计算流图机制  
研究GA对计算流图建模方式  
研究流图拓扑排序对显存占用的影响  
研究pytorch内存碎片问题  
研究olla如何将两个问题合在一起，多目标  
下次汇报，5.12

#### 给alexnet打上查看显存分配的断点 
<font color='red'> 未完成 </font> 


## 5.10任务
#### 重要：解决LeNet384MB问题及alexnet显存分配问题
<font color='red'> 未完成 </font>
#### 调研PyTorch计算图
<font color='LimeGreen'> 完成 </font>
#### 结合GA给出计算图建模思路
<font color='red'> 未完成 </font>
#### 调研OLLA如何做多目标
<font color='LimeGreen'> 完成 </font>：没有做多目标

## 5.11任务
#### 研究数据流图，写汇报文档
<font color='LimeGreen'> 完成 </font>
#### 调研tensor.to调用显存步骤
目前进度：pytorch源码，Tensor.cpp,TensorBase.h,TemsorAccessor.h记录tensor的size和strides信息,TensorOptions.cpp  
<font color='LimeGreen'> 完成 </font>
#### 补进度  


## 5.12任务
#### 看完OLLA实验部分，写汇报文档
<font color='LimeGreen'> 完成 </font>
#### 汇报进度，得到反馈
<font color='blue'> 暂缓 </font>
#### 看swapadvisor，学习GA建模方式
<font color='LimeGreen'> 完成 </font>
#### 建模
<font color='LimeGreen'> 5.31 完成 </font>


## 5.13任务
#### 整理知识碎片，写出3000字
<font color='LimeGreen'> 5.31 完成 </font>
#### 结合GA，神经网络建模
<font color='red'> 未完成 </font>

## 5.14任务
#### 建模
<font color='LimeGreen'> 5.31 完成 </font> 
#### 补论文进度至3000字
<font color='LimeGreen'> 5.31 完成 </font> 
#### torch.fx
<font color='LimeGreen'> 完成，仍需进一步了解 </font> 
## 5.15任务
#### 搞清楚torch.fx原理
<font color='LimeGreen'> 完成 </font>
#### 结合olla思考建模方式，DTR同步研究  
<font color='red'> 未完成 </font> 
#### 写1000字论文
<font color='LimeGreen'> 5.31 完成 </font> 
#### 给神经网络训练流程建立一个初步模型  
结点表示什么，边表示什么，有哪些属性，目标函数，约束条件  
<font color='red'> 未完成 </font> 

## 5.16任务
#### 建立初步模型，完成汇报文档
<font color='LimeGreen'> 5.31 完成 </font> 
#### 论文字数到3000字
<font color='LimeGreen'> 5.31 完成 </font> 

## 5.17任务
#### 跑通olla 
<font color='LimeGreen'> 完成，gutobi问题已解决 </font>
#### pycharm远程调试功能
<font color='LimeGreen'> 完成 </font>
#### 看OLLA代码，结合论文生成文档
<font color='LimeGreen'> 5.28 完成 </font> 

## 5.18任务
#### 在服务器上部署olla环境，激活gurobi
<font color='LimeGreen'> 完成 </font>
#### 运行olla
<font color='LimeGreen'> 完成 </font> 
#### 掌握benchmarks.py各种函数意思
<font color='LimeGreen'> 5.28完成 </font> 
#### 论文正文至2500字
<font color='LimeGreen'> 5.28完成 </font> 

## 5.19任务
#### 掌握benchmarks.py个函数意思
未完成，看到249行，生成计算图位置   
<font color='LimeGreen'> 5.28完成 </font>
#### 运行olla
<font color='LimeGreen'> 完成 </font> 

## 5.20任务
#### 继续看源码
<font color='LimeGreen'> 5.28完成 </font> 
#### 若能连接服务器，解决olla关于gurobi环境变量问题
仿照gurobi安装目录/examples/python下的范例,配置环境变量  
<font color='LimeGreen'> 完成，环境变量注释后程序仍可运行 </font> 
## 5.22任务
#### 弄清loadModel中所有函数的作用
<font color='LimeGreen'> 5.28完成 </font> 

## 5.23任务
#### 单个输出网络评估结果，通过流输出log日志
<font color='LimeGreen'> 完成 </font> 
#### 看完源码，输出理解文档
<font color='LimeGreen'> 5.28完成 </font> 

## 5.24任务
#### 写500字论文
<font color='LimeGreen'> 完成，5.26 </font>
#### 理解fx原理，生成文档
<font color='LimeGreen'> 完成 </font>
#### 理解olla源码，从`torch_graph_importer`236行
<font color='LimeGreen'> 5.28完成 </font>

## 5.25任务
#### 理解olla源码，生成汇报PPT
从`torch_graph_importer`236行  
<font color='red'> 未完成 </font>
#### 调研卷积优化算法，例如googlenet卷积优化策略
<font color='red'> 未完成 </font>

## 5.26任务
#### 汇报
<font color='LimeGreen'> 5.31完成 </font>
#### 源码
<font color='LimeGreen'> 5.28完成 </font>

#### 远程调试代码
<font color='LimeGreen'> 5.27总结： </font>
换用vscode远程调试程序，优点：不卡顿，方便备注；缺点：debug功能不如pycharm强大，一些变量类型识别不了，也无法在代码侧实时显示变量值

## 5.28任务
#### 从断点继续看源码，理解调整顺序的操作
benchmarks.py：664行  
<font color='LimeGreen'> 5.28总结：</font>初步看完源码中benchmarks的所有调用，
#### 写关于遗传算法论文500字
重点编码部分（建模）  
<font color='LimeGreen'> 5.31 完成 </font>3298 

## 5.29任务
#### 完成遗传算法编码
<font color='red'> 未完成 </font>
#### 撰写论文第三章理论部分1000字
<font color='LimeGreen'> 5.31完成 </font>
#### 找关于优化遗传算法的trick
<font color='red'> 未完成 </font>

## <font color='LimeGreen'> 5.31总结： </font>   
完成建模汇报，定下本周目标为写完第三章理论部分内容。  
#### 遗留问题：  
用GA解决问题，在哪些方面优于其他启发式算法，从效率和结果来看  
实验结果可以不与原文比对  
第二点做什么？老师不推荐修改算子，使用重计算尽量使优化目标与第一点相同，需要调研  

## 6.1任务
#### 撰写论文
推第三章内容，核心部分3.3节的建模，3.4节的求解。其次3.1，3.2
