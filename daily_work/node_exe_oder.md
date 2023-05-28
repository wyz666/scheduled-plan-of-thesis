fx图中节点遍历顺序，即pytorch运行alexnet模型的算子顺序：
args_1(inputs),param_2,param_3,...参数包含权重以及偏置  
conv,relu,pool,getitem,getitem_1,  
conv_1,relu_1,pool_1,getitem_2,getitem_3,  
conv_2,relu_2,conv_3,relu_3,   
conv_4,relu_4,pool_2,getitem_4,getitem_5, ... 
t 转置操作
addmm 矩阵相乘操作 

fx图节点会记录users，还会记录所有输入节点
