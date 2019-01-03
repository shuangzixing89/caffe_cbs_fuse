from src.rw_model import *
import numpy as np

##提取Conv+BatchNorm+Scale的名字和BatchNorm、Scale在Layer列表中的index
def getConvBNLayer(net):
    Layer = getNetLayer(net)
    Conv_BN_list = []
    BN_index = []
    for i in range(len(Layer)):
        ConvBN = []
        if (i+1)<len(Layer) and Layer[i].type == "Convolution" or Layer[i].type == 4 :
            if Layer[i + 1].type == "BatchNorm" and Layer[i + 2].type == "Scale":
                ConvBN.append(Layer[i].name)
                ConvBN.append(Layer[i + 1].name)
                ConvBN.append(Layer[i + 2].name)
                Layer[i].convolution_param.bias_term = True
                BN_index.append(i+1)
                BN_index.append(i+2)
                Conv_BN_list.append(ConvBN)
    return Conv_BN_list,BN_index

def getParam(name,model):
    ModelLayer = getModelLayer(model)
    for i in range(len(ModelLayer)):
        if ModelLayer[i].name == name:
            params = ModelLayer[i].blobs
            break
    return params,i


def disposeParams(Params):
    data,shape = np.array(Params.data),Params.shape.dim
    P_data = data.reshape(shape)
    return P_data


##修改参数，将Conv、BatchNorm、Scale的参数融合再存到Conv中
def modifyParam(ConvBNName,model):
    for CBN in ConvBNName:

        W_bc,conv_index = getParam(CBN[0],model)
        W_data = disposeParams(W_bc[0])

        bc_data = 0
        if len(W_bc) != 1:
            bc_data = disposeParams(W_bc[1])

        m_v,bn_index = getParam(CBN[1],model)
        m_data = disposeParams(m_v[0])
        v_data = disposeParams(m_v[1])
        s_bs,scale_index = getParam(CBN[2],model)
        s_data = disposeParams(s_bs[0])
        bs_data = disposeParams(s_bs[1])

        W_new = []
        for i in range(W_data.shape[0]):
            temp = W_data[i] * s_data[i] / np.sqrt(v_data[i] + 0.00001)
            W_new.append(temp)
        b_new = (bc_data - m_data) * s_data / np.sqrt(v_data + 0.00001) + bs_data


        model.layer[conv_index].blobs[0].data[:] = np.array(W_new).reshape(1,-1).tolist()[0]
        if len(W_bc) == 1:
            model.layer[scale_index].blobs[1].data[:] = np.array(b_new).reshape(1, -1).tolist()[0]
            model.layer[conv_index].blobs.extend([model.layer[scale_index].blobs[1]])


        else:
            model.layer[conv_index].blobs[1].data[:] = np.array(b_new).reshape(1,-1).tolist()[0]

    return model
