from src import rw_model as RWM
from src import conv_bn_scale_fuse as CBS
import sys,getopt

def usage():
    print("""
    python fuse_caffe.py prototxt_path caffemodel_path fuse_name [--fusepath=]
    """)


def main():
    if len(sys.argv) == 1:
        usage()
        sys.exit()
    fuse_path = "./"
    opts, args = getopt.getopt(sys.argv[4:],"h",["fusepath="])
    for op, value in opts:
        if op == "-h":
            usage()
            sys.exit()
        elif op == "--fusepath":
            fuse_path = str(value)
    prototxt_path = str(sys.argv[1])
    caffemodel_path = str(sys.argv[2])
    fuse_name = str(sys.argv[3])

    net = RWM.readPrototxt(prototxt_path)
    model = RWM.readCaffemodel(caffemodel_path)

    Conv_Bn_Scale_Name, CBSN_index = CBS.getConvBNLayer(net)
    CBS.delLayer(net, CBSN_index)
    CBS.savePrototxt(net, fuse_path+fuse_name+"_fuse")
    model = CBS.modifyParam(Conv_Bn_Scale_Name, model)
    CBS.saveCaffemodel(model,fuse_path+fuse_name+"_fuse")


if __name__ == '__main__':
    main()



