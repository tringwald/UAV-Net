from __future__ import division
from __future__ import print_function
import argparse
import caffe as c
import numpy as np
import sys
import json
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--initial-weights', type=str, default='final_arch_full_1x1.caffemodel')
parser.add_argument('--initial-proto', type=str, default='final_arch_test.prototxt')
parser.add_argument('--weights', type=str, required=False, help="Dont set this")
parser.add_argument('--proto', type=str, required=False, help="Dont set this")
parser.add_argument('--num-val-examples', type=int, default=289)
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--template', default='template.prototxt')
parser.add_argument('--mode', default='1x1', choices=['1x1', 'gconv'])
args = parser.parse_args()
c.set_device(0)
c.set_mode_gpu()


mapping = OrderedDict([('conv1', 64),
                       ('fire2/squeeze3x3', 16), ('fire2/expand1x1', 64), ('fire2/expand3x3', 64),
                       ('fire3/squeeze1x1', 16), ('fire3/expand1x1', 64), ('fire3/expand3x3', 64),
                       ('fire4/squeeze3x3', 32), ('fire4/expand1x1', 128), ('fire4/expand3x3', 128),
                       ('fire5/squeeze1x1', 32), ('fire5/expand1x1', 128), ('fire5/expand3x3', 128)
                       ])

cost_mapping = OrderedDict([('conv1', 3 * 3 * 3 * 936 * 624 / 4),
                            ('fire2/squeeze3x3', 64 * 3 * 3 * 468 * 312 / 4), ('fire2/expand1x1', 16 * 1 * 1 * 234 * 156),
                            ('fire2/expand3x3', 16 * 3 * 3 * 234 * 156),
                            ('fire3/squeeze1x1', 128 * 1 * 1 * 234 * 156), ('fire3/expand1x1', 16 * 1 * 1 * 234 * 156),
                            ('fire3/expand3x3', 16 * 3 * 3 * 234 * 156),
                            ('fire4/squeeze3x3', 128 * 3 * 3 * 234 * 156 / 4), ('fire4/expand1x1', 32 * 1 * 1 * 117 * 78),
                            ('fire4/expand3x3', 32 * 3 * 3 * 117 * 78),
                            ('fire5/squeeze1x1', 256 * 1 * 1 * 117 * 78), ('fire5/expand1x1', 32 * 1 * 1 * 117 * 78), ('fire5/expand3x3', 32 * 3 * 3 * 117 * 78)
                            ])

summary_template = OrderedDict([('conv1', 0),
                           ('fire2/squeeze3x3', 0), ('fire2/expand1x1', 0), ('fire2/expand3x3', 0),
                           ('fire3/squeeze1x1', 0), ('fire3/expand1x1', 0), ('fire3/expand3x3', 0),
                           ('fire4/squeeze3x3', 0), ('fire4/expand1x1', 0), ('fire4/expand3x3', 0),
                           ('fire5/squeeze1x1', 0), ('fire5/expand1x1', 0), ('fire5/expand3x3', 0)])


def queue_to_summary(del_q):
    delete_summary = summary_template.copy()
    for layer_name, count in del_q:
        delete_summary[layer_name] += count
    return delete_summary


def prune(args, delete_queue):
    net = c.Net(args.proto, args.weights, c.TEST)
    num_pos = 0
    num_found = 0

    if args.test:
        for i in range(args.num_val_examples):
            det_eval = net.forward()['detection_eval'].squeeze()
            if len(det_eval.shape) != 2:
                det_eval = np.expand_dims(det_eval, 0)
            num_pos += det_eval[0, :][2]
            num_found += sum(det_eval[1:, :][:, 3] == 1)

        return num_found / num_pos

    # Create the prototxt
    template = open(args.template).read()
    values = {}
    delete_summary = queue_to_summary(delete_queue)
    for k, v in mapping.iteritems():
        values[k] = v - delete_summary[k]
    open('target.prototxt', 'w+').write(template % values)
    target_net = c.Net('target.prototxt', c.TEST)

    def judge_function(name, w, num_pruning):
        num_filters = w.shape[0]
        def l1_norm(x):
            return np.abs(x).sum()
        weights = [(i, w[i, :, :, :]) for i in range(num_filters)]
        weights = sorted(weights, key=lambda x: l1_norm(x[1]))
        return list(map(lambda x: x[0], weights))[:num_pruning]

    # Copy network weights as inplace modification doesnt work
    tmp_net = OrderedDict()
    for layer, weights in net.params.iteritems():
        try:
            if 'mbox' in layer and args.mode == 'gconv':
                num_filters = weights[0].data.shape[0]
                tmp_net[layer] = [[weights[0].data[:num_filters // 2], weights[0].data[num_filters // 2:]], weights[1].data[:]]
            else:
                tmp_net[layer] = [weights[0].data[:], weights[1].data[:]]
        except:
            tmp_net[layer] = [weights[0].data[:]]

    from collections import defaultdict
    already_deleted = defaultdict(int)
    # Iter over the delete list
    for layer, num_pruning in delete_queue:
        if 'mbox' in layer or 'norm' in layer or layer not in delete_summary or delete_summary[layer] == 0:
            continue
        params = tmp_net[layer]
        weights, bias = params
        remove_idx = judge_function(layer, weights, num_pruning)

        # Remove filters
        tmp_net[layer][0] = np.delete(weights, remove_idx, axis=0)
        tmp_net[layer][1] = np.delete(bias, remove_idx, axis=0)
        already_deleted[layer] += num_pruning

        # Remove channels of next filter
        if layer == 'conv1':
            target = 'fire2/squeeze3x3'
            tmp_net[target][0] = np.delete(tmp_net[target][0], remove_idx, axis=1)
        else:
            fire_module = int(layer[4])

            if 'squeeze' in layer:
                target = 'fire{}/expand3x3'.format(fire_module)
                tmp_net[target][0] = np.delete(tmp_net[target][0], remove_idx, axis=1)

                target = 'fire{}/expand1x1'.format(fire_module)
                tmp_net[target][0] = np.delete(tmp_net[target][0], remove_idx, axis=1)
            elif 'expand' in layer:
                if fire_module == 5:
                    # Prune additional layers
                    target = 'fire5/concat_norm'
                    offset = 0 if '1x1' in layer else net.params[target][0].shape[0] / 2. - already_deleted.get(layer.replace('3x3', '1x1'), 0)
                    tmp_net[target][0] = np.delete(tmp_net[target][0], np.array(remove_idx) + offset, axis=0)

                    if args.mode == '1x1':
                        target = 'fire5/concat_norm_mbox_loc'
                        tmp_net[target][0] = np.delete(tmp_net[target][0], np.array(remove_idx) + offset, axis=1)
                        target = 'fire5/concat_norm_mbox_conf'
                        tmp_net[target][0] = np.delete(tmp_net[target][0], np.array(remove_idx) + offset, axis=1)
                        continue
                    elif args.mode == 'gconv':
                        offset = 0

                        target = 'fire5/concat_norm_mbox_loc'
                        if '1x1' in layer:
                            tmp_net[target][0][0] = np.delete(tmp_net[target][0][0][:], np.array(remove_idx) + offset, axis=1)
                        else:
                            tmp_net[target][0][1] = np.delete(tmp_net[target][0][1][:], np.array(remove_idx) + offset, axis=1)

                        target = 'fire5/concat_norm_mbox_conf'
                        if '1x1' in layer:
                            tmp_net[target][0][0] = np.delete(tmp_net[target][0][0][:], np.array(remove_idx) + offset, axis=1)
                        else:
                            tmp_net[target][0][1] = np.delete(tmp_net[target][0][1][:], np.array(remove_idx) + offset, axis=1)

                        continue
                else:  # Other fire modules
                    next_squeeze = 'fire{}/squeeze1x1'.format(fire_module + 1)
                    if next_squeeze not in tmp_net:
                        next_squeeze = 'fire{}/squeeze3x3'.format(fire_module + 1)

                    if '1x1' in layer:
                        offset = 0
                    else:
                        offset = net.params[next_squeeze][0].shape[1] / 2. - already_deleted.get(layer.replace('3x3', '1x1'), 0)
                    tmp_net[next_squeeze][0] = np.delete(tmp_net[next_squeeze][0], np.array(remove_idx) + offset, axis=1)
            else:
                raise AssertionError

    for layer, params in tmp_net.iteritems():
        if len(params) > 1:
            weights, bias = params
            if isinstance(weights, list):
                target_net.params[layer][0].data[:] = np.vstack(weights)
            else:
                target_net.params[layer][0].data[:] = params[0]
            target_net.params[layer][1].data[:] = params[1]
        else:
            target_net.params[layer][0].data[:] = params[0]
    target_net.save('target_weights.caffemodel')
    del net
    del target_net


def arch_scoring(prec, speedup):
    return prec


def make_net():
    """ Use this function to generate the target prototxt and weight file by passing a delete queue to the prune function.
    Below are the different delete queues that generate UAV-Nets.
    """
    total_num_filters = sum(mapping.values())
    # 0.9892871858261227
    uavnet_1x1_4filters75_tp = [('fire2/squeeze3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire4/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire2/expand3x3', 4), ('fire2/expand1x1', 4), ('fire2/expand1x1', 4), ('fire2/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire4/expand1x1', 4), ('fire5/expand1x1', 4), ('fire2/expand3x3', 4), ('fire2/expand1x1', 4), ('fire2/expand1x1', 4), ('fire5/expand1x1', 4), ('fire4/expand1x1', 4), ('fire2/expand3x3', 4), ('fire2/expand3x3', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('conv1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('conv1', 4), ('conv1', 4), ('fire3/expand1x1', 4), ('fire4/expand1x1', 4), ('fire2/expand1x1', 4), ('fire3/expand1x1', 4), ('fire3/expand1x1', 4), ('conv1', 4), ('fire2/expand1x1', 4), ('fire5/expand1x1', 4), ('fire3/expand1x1', 4), ('fire4/expand1x1', 4), ('fire2/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4)]
    print("Filter percentage", (total_num_filters - len(uavnet_1x1_4filters75_tp) * 4) / total_num_filters)
    # 0.9373712402142563
    uavnet_1x1_4filters50_tp = [('fire2/squeeze3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire4/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire2/expand3x3', 4), ('fire2/expand1x1', 4), ('fire2/expand1x1', 4), ('fire2/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire4/expand1x1', 4), ('fire5/expand1x1', 4), ('fire2/expand3x3', 4), ('fire2/expand1x1', 4), ('fire2/expand1x1', 4), ('fire5/expand1x1', 4), ('fire4/expand1x1', 4), ('fire2/expand3x3', 4), ('fire2/expand3x3', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('conv1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('conv1', 4), ('conv1', 4), ('fire3/expand1x1', 4), ('fire4/expand1x1', 4), ('fire2/expand1x1', 4), ('fire3/expand1x1', 4), ('fire3/expand1x1', 4), ('conv1', 4), ('fire2/expand1x1', 4), ('fire5/expand1x1', 4), ('fire3/expand1x1', 4), ('fire4/expand1x1', 4), ('fire2/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('conv1', 4), ('fire4/expand1x1', 4), ('conv1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire2/expand1x1', 4), ('fire3/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire5/expand3x3', 4), ('fire2/expand1x1', 4), ('fire4/expand3x3', 4), ('fire2/expand3x3', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire2/expand3x3', 4), ('fire3/expand1x1', 4), ('fire3/expand1x1', 4), ('fire3/expand1x1', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand1x1', 4), ('fire2/expand3x3', 4), ('fire2/expand3x3', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire3/expand1x1', 4), ('fire4/expand1x1', 4), ('fire3/expand3x3', 4), ('fire3/expand3x3', 4), ('fire5/expand3x3', 4), ('fire4/expand1x1', 4), ('fire3/expand1x1', 4), ('fire4/expand1x1', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/squeeze1x1', 4), ('fire2/expand3x3', 4), ('fire2/expand3x3', 4)]
    print("Filter percentage", (total_num_filters - len(uavnet_1x1_4filters50_tp) * 4) / total_num_filters)
    # 0.5515039142974866
    uavnet_1x1_4filters25_tp = [('fire2/squeeze3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire4/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire2/expand3x3', 4), ('fire2/expand1x1', 4), ('fire2/expand1x1', 4), ('fire2/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire4/expand1x1', 4), ('fire5/expand1x1', 4), ('fire2/expand3x3', 4), ('fire2/expand1x1', 4), ('fire2/expand1x1', 4), ('fire5/expand1x1', 4), ('fire4/expand1x1', 4), ('fire2/expand3x3', 4), ('fire2/expand3x3', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('conv1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('conv1', 4), ('conv1', 4), ('fire3/expand1x1', 4), ('fire4/expand1x1', 4), ('fire2/expand1x1', 4), ('fire3/expand1x1', 4), ('fire3/expand1x1', 4), ('conv1', 4), ('fire2/expand1x1', 4), ('fire5/expand1x1', 4), ('fire3/expand1x1', 4), ('fire4/expand1x1', 4), ('fire2/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('conv1', 4), ('fire4/expand1x1', 4), ('conv1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire2/expand1x1', 4), ('fire3/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire5/expand3x3', 4), ('fire2/expand1x1', 4), ('fire4/expand3x3', 4), ('fire2/expand3x3', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire2/expand3x3', 4), ('fire3/expand1x1', 4), ('fire3/expand1x1', 4), ('fire3/expand1x1', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand1x1', 4), ('fire2/expand3x3', 4), ('fire2/expand3x3', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire3/expand1x1', 4), ('fire4/expand1x1', 4), ('fire3/expand3x3', 4), ('fire3/expand3x3', 4), ('fire5/expand3x3', 4), ('fire4/expand1x1', 4), ('fire3/expand1x1', 4), ('fire4/expand1x1', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/squeeze1x1', 4), ('fire2/expand3x3', 4), ('fire2/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/squeeze1x1', 4), ('fire5/squeeze1x1', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire2/expand3x3', 4), ('fire2/expand3x3', 4), ('fire2/expand3x3', 4), ('fire5/squeeze1x1', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/squeeze1x1', 4), ('fire2/expand3x3', 4), ('fire2/expand3x3', 4), ('conv1', 4), ('conv1', 4), ('conv1', 4), ('conv1', 4), ('conv1', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire2/squeeze3x3', 4), ('fire2/expand1x1', 4), ('fire5/squeeze1x1', 4), ('fire3/expand3x3', 4), ('fire3/expand3x3', 4), ('fire4/squeeze3x3', 4), ('fire5/expand3x3', 4), ('fire3/expand3x3', 4), ('fire3/expand1x1', 4), ('fire4/expand3x3', 4), ('fire3/expand1x1', 4), ('fire3/expand1x1', 4), ('fire3/expand1x1', 4), ('fire2/expand1x1', 4), ('fire5/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire3/expand3x3', 4)]
    print("Filter percentage", (total_num_filters - len(uavnet_1x1_4filters25_tp) * 4) / total_num_filters)
    # 0.28594973217964564
    uavnet_1x1_4filters15_tp = [('fire2/squeeze3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire4/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire2/expand3x3', 4), ('fire2/expand1x1', 4), ('fire2/expand1x1', 4), ('fire2/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire4/expand1x1', 4), ('fire5/expand1x1', 4), ('fire2/expand3x3', 4), ('fire2/expand1x1', 4), ('fire2/expand1x1', 4), ('fire5/expand1x1', 4), ('fire4/expand1x1', 4), ('fire2/expand3x3', 4), ('fire2/expand3x3', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('conv1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('conv1', 4), ('conv1', 4), ('fire3/expand1x1', 4), ('fire4/expand1x1', 4), ('fire2/expand1x1', 4), ('fire3/expand1x1', 4), ('fire3/expand1x1', 4), ('conv1', 4), ('fire2/expand1x1', 4), ('fire5/expand1x1', 4), ('fire3/expand1x1', 4), ('fire4/expand1x1', 4), ('fire2/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('conv1', 4), ('fire4/expand1x1', 4), ('conv1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire2/expand1x1', 4), ('fire3/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire5/expand3x3', 4), ('fire2/expand1x1', 4), ('fire4/expand3x3', 4), ('fire2/expand3x3', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire2/expand3x3', 4), ('fire3/expand1x1', 4), ('fire3/expand1x1', 4), ('fire3/expand1x1', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand1x1', 4), ('fire2/expand3x3', 4), ('fire2/expand3x3', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire3/expand1x1', 4), ('fire4/expand1x1', 4), ('fire3/expand3x3', 4), ('fire3/expand3x3', 4), ('fire5/expand3x3', 4), ('fire4/expand1x1', 4), ('fire3/expand1x1', 4), ('fire4/expand1x1', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/squeeze1x1', 4), ('fire2/expand3x3', 4), ('fire2/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/squeeze1x1', 4), ('fire5/squeeze1x1', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire2/expand3x3', 4), ('fire2/expand3x3', 4), ('fire2/expand3x3', 4), ('fire5/squeeze1x1', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/squeeze1x1', 4), ('fire2/expand3x3', 4), ('fire2/expand3x3', 4), ('conv1', 4), ('conv1', 4), ('conv1', 4), ('conv1', 4), ('conv1', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire2/squeeze3x3', 4), ('fire2/expand1x1', 4), ('fire5/squeeze1x1', 4), ('fire3/expand3x3', 4), ('fire3/expand3x3', 4), ('fire4/squeeze3x3', 4), ('fire5/expand3x3', 4), ('fire3/expand3x3', 4), ('fire3/expand1x1', 4), ('fire4/expand3x3', 4), ('fire3/expand1x1', 4), ('fire3/expand1x1', 4), ('fire3/expand1x1', 4), ('fire2/expand1x1', 4), ('fire5/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire3/expand3x3', 4), ('conv1', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire3/squeeze1x1', 4), ('fire3/expand1x1', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire2/expand1x1', 4), ('fire3/expand3x3', 4), ('fire3/expand3x3', 4), ('fire2/expand1x1', 4), ('fire3/expand3x3', 4), ('fire3/expand3x3', 4), ('fire3/expand3x3', 4), ('conv1', 4), ('fire4/expand3x3', 4), ('conv1', 4), ('fire4/expand3x3', 4), ('fire4/squeeze3x3', 4), ('fire2/squeeze3x3', 4), ('fire5/expand3x3', 4)]
    print("Filter percentage", (total_num_filters - len(uavnet_1x1_4filters15_tp) * 4) / total_num_filters)
    # 0.17511330861145447
    uavnet_1x1_4filters075_tp = [('fire2/squeeze3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire4/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire2/expand3x3', 4), ('fire2/expand1x1', 4), ('fire2/expand1x1', 4), ('fire2/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire4/expand1x1', 4), ('fire5/expand1x1', 4), ('fire2/expand3x3', 4), ('fire2/expand1x1', 4), ('fire2/expand1x1', 4), ('fire5/expand1x1', 4), ('fire4/expand1x1', 4), ('fire2/expand3x3', 4), ('fire2/expand3x3', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('conv1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('conv1', 4), ('conv1', 4), ('fire3/expand1x1', 4), ('fire4/expand1x1', 4), ('fire2/expand1x1', 4), ('fire3/expand1x1', 4), ('fire3/expand1x1', 4), ('conv1', 4), ('fire2/expand1x1', 4), ('fire5/expand1x1', 4), ('fire3/expand1x1', 4), ('fire4/expand1x1', 4), ('fire2/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire5/expand1x1', 4), ('fire5/expand1x1', 4), ('conv1', 4), ('fire4/expand1x1', 4), ('conv1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire2/expand1x1', 4), ('fire3/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4),('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire5/expand3x3', 4), ('fire2/expand1x1', 4), ('fire4/expand3x3', 4), ('fire2/expand3x3', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire2/expand3x3', 4), ('fire3/expand1x1', 4), ('fire3/expand1x1', 4), ('fire3/expand1x1', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand1x1', 4), ('fire2/expand3x3', 4), ('fire2/expand3x3', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire4/expand1x1', 4), ('fire3/expand1x1', 4), ('fire4/expand1x1', 4), ('fire3/expand3x3', 4), ('fire3/expand3x3', 4), ('fire5/expand3x3', 4), ('fire4/expand1x1', 4), ('fire3/expand1x1', 4), ('fire4/expand1x1', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/squeeze1x1', 4), ('fire2/expand3x3', 4), ('fire2/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/squeeze1x1', 4), ('fire5/squeeze1x1', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire2/expand3x3', 4), ('fire2/expand3x3', 4), ('fire2/expand3x3', 4), ('fire5/squeeze1x1', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire5/squeeze1x1', 4), ('fire2/expand3x3', 4), ('fire2/expand3x3', 4), ('conv1', 4), ('conv1', 4), ('conv1', 4), ('conv1', 4), ('conv1', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire2/squeeze3x3', 4), ('fire2/expand1x1', 4), ('fire5/squeeze1x1', 4), ('fire3/expand3x3', 4), ('fire3/expand3x3', 4), ('fire4/squeeze3x3', 4), ('fire5/expand3x3', 4), ('fire3/expand3x3', 4), ('fire3/expand1x1', 4), ('fire4/expand3x3', 4), ('fire3/expand1x1', 4), ('fire3/expand1x1', 4), ('fire3/expand1x1', 4), ('fire2/expand1x1', 4), ('fire5/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire3/expand3x3', 4), ('conv1', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire3/squeeze1x1', 4), ('fire3/expand1x1', 4), ('fire5/expand3x3', 4), ('fire5/expand3x3', 4), ('fire2/expand1x1', 4), ('fire3/expand3x3', 4), ('fire3/expand3x3', 4), ('fire2/expand1x1', 4), ('fire3/expand3x3', 4), ('fire3/expand3x3', 4), ('fire3/expand3x3', 4), ('conv1', 4), ('fire4/expand3x3', 4), ('conv1', 4), ('fire4/expand3x3', 4), ('fire4/squeeze3x3', 4), ('fire2/squeeze3x3', 4), ('fire5/expand3x3', 4), ('fire2/expand1x1', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('conv1', 4), ('fire3/expand3x3', 4), ('fire3/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire4/squeeze3x3', 4), ('fire4/expand3x3', 4), ('fire4/expand3x3', 4), ('fire5/squeeze1x1', 4), ('fire4/expand3x3', 4), ('fire3/squeeze1x1', 4), ('fire3/expand3x3', 4), ('fire4/squeeze3x3', 4), ('fire3/expand3x3', 4)]
    print("Filter percentage", (total_num_filters - len(uavnet_1x1_4filters075_tp) * 4) / total_num_filters)

    current = uavnet_1x1_4filters75_tp

    args.test = False
    args.weights = args.initial_weights
    args.proto = args.initial_proto
    # Pass the delete queue as 2nd parameter to generate the network
    prune(args, current)

    args.test = True
    args.weights = 'target_weights.caffemodel'
    args.proto = 'target.prototxt'
    percentage = prune(args, [])
    print("Filter percentage", (total_num_filters - len(current) * 4) / total_num_filters)
    print(percentage)
    sys.exit(0)


if __name__ == '__main__':
    np.random.seed(0)
    delete_queue = []

    out_stream = open('log.txt', 'a+')
    args.test = True
    args.weights = args.initial_weights
    args.proto = args.initial_proto
    initial_arch_percentage = prune(args, [])
    print("Initial", initial_arch_percentage)

    current_arch_score = 1.00
    current_arch_pruned_percentage = 1.00
    current_arch_deleted_percentage = 1.00
    while current_arch_deleted_percentage >= 0.05:
        current_iter_archs = []

        # Reversed to prefer later layers over initial ones
        for layer in reversed(summary_template.keys()):
            print("Pruning", layer)
            current_queue = delete_queue[:]
            args.test = False
            args.weights = args.initial_weights
            args.proto = args.initial_proto

            # Generate a new arch
            current_queue.append((layer, 4))

            # For gconv only prune one side and then apply the symmetry constraint
            if args.mode == 'gconv' and 'fire5/expand3x3' in layer:
                continue
            if args.mode == 'gconv' and 'fire5/expand1x1' in layer:
                current_queue.append((layer.replace('1x1', '3x3'), 4))

            current_arch = queue_to_summary(current_queue)
            # At least 4 filters should remain, skip otherwise
            if current_arch[layer] == mapping[layer]:
                print("Skipping", layer)
                continue
            prune(args, current_queue)

            # Test it
            args.test = True
            args.weights = 'target_weights.caffemodel'
            args.proto = 'target.prototxt'
            percentage = prune(args, [])
            num_deleted = sum(current_arch.values())
            total_num_filters = sum(mapping.values())

            full_cost = 0
            pruned_cost = 0
            for k, v in mapping.iteritems():
                full_cost += v * cost_mapping[k]
                pruned_cost += (v - current_arch.get(k, 0)) * cost_mapping[k]
            current_iter_archs.append((arch_scoring(percentage, 1 - (pruned_cost / full_cost)), current_queue, pruned_cost / full_cost,
                                       (total_num_filters - num_deleted) / total_num_filters))

            print(percentage, initial_arch_percentage - percentage, pruned_cost / full_cost, pruned_cost, full_cost, num_deleted,
                  json.dumps([(k, mapping[k] - v) for k, v in current_arch.iteritems()]),
                  file=out_stream)
            print(percentage, initial_arch_percentage - percentage, pruned_cost / full_cost, pruned_cost, full_cost, num_deleted,
                  json.dumps([(k, mapping[k] - v) for k, v in current_arch.iteritems()]))

        # Choose currently best arch as master
        metric, best_queue, cost_percentage, deleted_percentage = list(sorted(current_iter_archs, reverse=True, key=lambda x: x[0]))[0]
        delete_queue = best_queue[:]
        current_arch_score = metric
        current_arch_deleted_percentage = deleted_percentage
        print("\n\nChosen", metric, delete_queue)
        print("Chosen", metric, deleted_percentage, delete_queue, "\n\n", file=out_stream)

        print("Summary", queue_to_summary(delete_queue))
        print("Pruned layer", delete_queue[-1][0], "\n\n")
        print("Stats: del percentage: ", current_arch_deleted_percentage, "cost: ", cost_percentage)

