## 简介
torch.fx是用于捕获和转换神经网络结构的纯python系统。  
动机：pytorch采用动态图给神经网络用户带来便利，但是分析和优化网络结构需要静态图。动态计算图适用于模型开发调试，静态计算图适用于模型部署调优。  
特性：  
1. 采用追踪方式捕获神经网络结构-运行模型，记录所有运算和调用，与jit.trace（must provide example inputs）不同无需对输入进行特例化；  
2. IR(intermediate representation)只有6条指令，对应6种节点类型，不支持动态控制流，不支持可变状态和别名；  
3. 根据IR自动生成可运行的python代码

## 源码解析
### `symbolic_trace`函数
首先初始化`Tracer`类实例`tracer`，然后调用`tracer.tracer`追踪入参`root`（实例`m`，即模型）返回fx计算图`graph`。获取`root`的名字`name`。最后函数返回的是一个用`tracer.root, graph, name`初始化类`GraphModule`的实例`gm`。

### `trace`函数
初始化被追踪的模型和函数：如果传入的是模型，将模型赋值给`self.root`，默认追踪模型的`self.traced_func_name`方法，即`forward`方法，并赋值给`fn`；如果传入的是函数，会初始化一个模型，用来存储函数访问的属性。  
初始化计算图`self.graph`用来插入追踪的节点。  
构建张量到属性名映射：把张量全存到字典里，方便查找   
创建参数节点：首先断言`fn`必须是函数类型，获得其全局变量，之后会用于函数和方法的Monkey Patch。接着调用`self.create_args_for_root`方法，为`fn`的参数创造计算图节点。  

**处理模型输入，生成`placeholder`类型的Proxy** `create_args_for_root`方法：先对`fn`做了`unwrap`，拿到`fn`的代码对象`co`。`co.co_argcount`和`co.co_kwonlyargcount`算出`fn`入参数量`total_args`。判断`root`是否为模型，是的话加到`args: List[Any]`中。`co.co_varnames`记录了函数内所有局部变量的名字，入参是局部变量，前`total_args`个就是入参名，赋值给`arg_names`。遍历入参名，为每个入参调用`proxy_placeholder`函数，返回是调用`self.create_proxy`创建的`placeholder`类型的Proxy实例，将该实例加入到`args`中。`create_proxy`方法用入参创建一个新`Node`实例`node`，调用`self.proxy`方法创建一个新的`Proxy`实例`proxy`（包含一个`Node`和一个`Tracer`）。综上，`create_args_for_root`方法遍历`fn`的入参，为除`self`之外的参数生成Proxy。例如追踪的是`forward(self, x)`，那么返回的`args`包含两个元素，第一个是模型本身`self`，第二个是`Proxy(x)`。  

monkey patch：fx在追踪时，并不想调用真正的函数，而是想生成计算图节点记录调用函数的行为。这时候就用到monkey patch ，动态（运行时）的属性替换。追踪时，将函数的`__call__`方法替换成生成描述调用的计算图节点，返回`Proxy`类型，追踪完成把原来的`__call__`方法替换回去。fx 通过 `with _Patcher() as patcher:` 这样一个上下文管理实现这样的功能，其中patch方法实现替换并将原函数打包成元组`_PatchedFnSetItem`保存在`patches_made`中，`__exit__`方法中调用`patches_made`中所有元素的`revert`方法进行还原。  

**处理模型参数，生成`get_attr`类型的Proxy**`Module.__getattr__`的patch。`module_getattr_wrapper`方法首先拿到模型权重参数，后调用`self.getattr`方法：当参数是`Parameter`类型时，返回Proxy并存储在`parameter_proxy_cache`中；计算图默认不插入Tensor类型的参数buffer，直接返回参数本身，也可打开`self.proxy_buffer_attributes`插入buffer。  
**处理函数，生成`call_function`类型的Proxy**`operator`的patch。python内置函数，如加减乘除，都是模型参数和输入调用的方法，而在追踪时这些参数和输入都被包装成了`Proxy`，fx修改`Proxy`提供的魔法方法`magic_method`，将内置函数的实现替换成生成`call_function`类型的`Proxy`。  
torch内置函数，如`torch.topk`、`torch.sum`，都是`Tensor`调用`__torch_function__`方法。fx自定义了`Proxy`的`__torch_function__`方法供追踪时调用，返回`call_function`类型的`Proxy`。  
自定义函数和库函数，如len，如果用户想自定义计算图的节点，fx提供了一个`wrap`接口。该接口查看函数是否有计算图的节点作为输入，有就返回`call_function`类型的`Proxy`，没有就执行一下原函数，不作为节点。  
模块的patch。Trcer初始化函数包含一个参数`autowrap_modules`，fx默认自动patch了math模块，将math模块中对外提供的函数记录下来，然后进行替换。  
`Module.__call__`的patch。如果模块不是叶子模块，则递归进入模块内部追踪；如果是叶子模块，则返回一个`call_moudle`类型的`Proxy`。  
创建output节点