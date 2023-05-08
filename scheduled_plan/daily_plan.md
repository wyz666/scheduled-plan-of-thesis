
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
<font color='red'> 5.5晚注: </font>需要再找一篇交叉比较<font color='LimeGreen'> 5.7已解决 </font>  


#### 快速过一遍老师的论文
进行总结  
<font color='red'> 未完成 </font>

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
<font color='red'> 未完成 </font>

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
<font color='red'> 未完成 </font>

#### 调研遗传算法解决工作流问题，整理出汇报文档
<font color='red'> 未完成 </font>

## 5.9任务

####调研遗传算法解决工作流问题，整理出汇报文档

#### 继续看pytorch显存调度

#### 汇报进度，得到反馈

#### 给alexnet打上查看显存分配的断点