# caffe模型中conv层和bn、scale层的融合
## 原因  
在神经网络训练的过程中，BN层能够加速网络收敛，并且能够控制过拟合。不过这样也增加了一些运算和参数，在推理过程中，我们可以通过将BN层与卷积层的参数融合，来减少运算，并且为模型稍稍的瘦一下身。  
## 依赖  
- protobuf  
- numpy  
## 使用  
```
 python fuse_caffe.py prototxt_path caffemodel_path fuse_name [--fusepath=]
```
## 公式  
$$ X_{bn} = \frac{s(X - m)}{\sqrt{\sigma + \epsilon}} + b_{bn}$$  
$$ X_{conv} = X * W + b_{conv} $$  

由此推得：
$$ W' = W\frac{s}{\sqrt{\sigma + \epsilon}}$$
$$ b' = (b_{conv} - m)\frac{s}{\sqrt{\sigma + \epsilon}} + b_{bn}$$
