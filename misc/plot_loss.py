from __future__ import print_function
import json
import numpy as np
from sklearn.cluster import KMeans
from pprint import pprint
from matplotlib import pyplot as plt
from scipy import stats
from math import *
import numpy as np
import argparse


plt.rcParams['grid.linestyle'] = "-"
plt.rcParams['grid.color'] = 'lightgrey'
plt.rcParams['axes.axisbelow'] = True
plt.rcParams.update({'font.size': 20})
plt.grid()
#plt.style.use(['seaborn-pastel'])

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return a[:n-1] + (ret[n - 1:] / n).tolist()


parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str)
parser.add_argument('--smooth', default=False, action='store_true')
parser.add_argument('--draw', action='store_true', default=False)
args = parser.parse_args()

iters = []
losses = []

ap_iters = []
aps = []

MAX_AP = -1
for line in open(args.log, 'r').read().splitlines():
    if ', loss' in line:
        parts = line.split(' ')
        parts = list(filter(lambda x: x != '' and x != ' ', parts))
        try:
            iter = int(parts[5][:-1])
            if iter % 20 == 0:
                iters.append(iter)
                losses.append(float(parts[8]))
            else:
                continue
        except:
            print("Cant parse:", line)

    elif 'class1' in line:
        ap = float(line.split(' ')[-1]) * 100
        aps.append(ap)
        iter = iters[-1] if len(iters) > 0 else 0
        ap_iters.append(iter)

    if not args.draw and ap_iters and (ap_iters[-1] % 2500 == 0 or (ap_iters[-1] + 20) % 2500 == 0):
        # Flush out
        print('==> AP {:05d}: {:2.2f} (to max {:2.2f}), Loss {:2.2f} (over {})'.format(ap_iters[-1], aps[-1], aps[-1] - MAX_AP, 0 if ap_iters[-1] == 0 else np.average(losses), len(losses)))
        # Reset
        if aps[-1] > MAX_AP:
            MAX_AP = aps[-1]
        iters, losses, ap_iters, aps = [], [], [], []

if args.draw:
    # AP stats
    last = 0.
    _max = 0
    for i, ap in zip(ap_iters, aps):
        print("AP Iteration {:05d}: {:2.2f} (to last: {:02.3f}, to max: {:02.3f})".format(i, ap, ap - last, ap - _max))
        last = ap
        if ap > _max:
            _max = ap

    print('-'*100)

    # Loss stats
    NUM = 125
    accum = []
    _last = 100
    for i, loss in zip(iters, losses):
        accum.append(loss)

        if len(accum) == NUM:
            avg = np.average(accum)
            print("Loss Iteration {:05d}: {:2.2f} ({:2.2f})".format(i, avg, _last - avg))
            _last = avg
            accum = []

    print('Remainder', '-'*100)
    avg = np.average(accum)
    print("Loss Iteration {:05d}: {:2.2f} ({:2.2f})".format(i, avg, _last - avg))

    if args.smooth:
        losses = moving_average(losses, n=10)
    fig = plt.gcf()

    ax1 = plt.gca()
    ax2 = ax1.twinx()
    plt.axis('normal')

    ln1 = ax1.plot(iters, losses, 'o-', color='steelblue', label='Loss')
    ln2 = ax2.plot(ap_iters, aps, 'o-', color='orange', label='AP')

    ax1.set_ylabel("Training loss", color='steelblue')
    ax1.set_yticks(np.arange(0, 20, 1))
    ax2.set_ylabel("Validation AP", color='orange')

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='center right')

    ax1.grid(True)
    ax1.set_xlabel('Iterations')
    plt.show()