from __future__ import print_function
import caffe as c
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--proto', type=str)
args = parser.parse_args()

net = c.Net(args.proto, c.TEST)


ctr = 0
for layer_name in net.params.keys():
    print(layer_name)
    layer = net.params[layer_name]

    weight_num = reduce(lambda x, y: x*y, layer[0].data.shape)
    ctr += weight_num
    print(layer[0].data.shape, weight_num)
    if len(layer) > 1:
        bias_num = reduce(lambda x, y: x*y, layer[1].data.shape)
        ctr += bias_num
        print(layer[1].data.shape, bias_num)

print('-'*100)
print(ctr)