
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
<font color='blue'> 暂缓 </font>  
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
<font color='red'> 未完成 </font>


## 5.13任务
#### 整理知识碎片，写出3000字
<font color='red'> 未完成 </font>
#### 结合GA，神经网络建模
<font color='red'> 未完成 </font>

## 5.14任务
#### 建模
<font color='red'> 未完成 </font> 
#### 补论文进度至3000字
<font color='red'> 未完成 </font> 
#### torch.fx
<font color='red'> 未完成 </font> 
## 5.15任务
#### 搞清楚torch.fx原理
<font color='LimeGreen'> 完成 </font>
#### 结合olla思考建模方式，DTR同步研究  
<font color='red'> 未完成 </font> 
#### 写1000字论文
<font color='red'> 未完成450/1000 </font> 
#### 给神经网络训练流程建立一个初步模型  
结点表示什么，边表示什么，有哪些属性，目标函数，约束条件  
<font color='red'> 未完成 </font> 

## 5.16任务
#### 建立初步模型，完成汇报文档
#### 论文字数到3000字