from src.caffe_pb2 import *
from google.protobuf import text_format


#######################################prototxt############################################

##读取prototxt
def readPrototxt(Orign_Prototxt_Path):
    net = NetParameter()
    text_format.Merge(open(Orign_Prototxt_Path).read(), net)
    return net

##获取prototxt中layer
def getNetLayer(net):
    if len(net.layer):
        Layer = net.layer
    elif len(net.layers):
        Layer = net.layers
    return Layer

##保存修改后的prototxt
def savePrototxt(net,Fuse_Model_Path):
    with open(Fuse_Model_Path+".prototxt","w") as f:
        print(net,file=f)

##根据index删除prototxt中layer
def delLayer(net,index):
    Layer = getNetLayer(net)
    i = 0
    for a in index:
        del Layer[a - i]
        i = i + 1

##获取layer信息
def getLayerInfo(net):
    if len(net.layer):
        Layer = net.layer
        print("一共",len(net.layer),"个layer")
    elif len(net.layers):
        Layer = net.layers
        print("一共",len(net.layers),"个layer")
    layer_type = [x.type for x in Layer]
    print("按顺序layer type：",layer_type)
    print("一共这几种layer type：",list(set(layer_type)))
    print("\n")
    for type in list(set(layer_type)):
        name = [layer.name for layer in net.layer if layer.type == type]
        print(type,"共",len(name),"个")
        print(type," name:",name)
        print("\n")

########################################caffemodel#########################################

##读取caffemodel
def readCaffemodel(Orign_Caffemodel_Path):
    model = NetParameter()
    with open(Orign_Caffemodel_Path, 'rb') as f:
        model.ParseFromString(f.read())
    return model


##获取caffemodel中layer
def getModelLayer(model):
    if len(model.layer):
        Layer = model.layer
    elif len(model.layers):
        Layer = model.layers
    return Layer

##保存修改后的caffemodel
def saveCaffemodel(model,Fuse_Model_Path):
    data = model.SerializeToString()
    with open(Fuse_Model_Path+".caffemodel","wb") as f:
        f.write(data)






