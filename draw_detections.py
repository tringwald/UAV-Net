# encoding=utf8
from __future__ import print_function, division
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['GLOG_minloglevel'] = '3'
import argparse
import cv2
import numpy as np
import os.path as osp
import time
import caffe
import glob


class CaffeDetection:
    def __init__(self, model_def, model_weights, gpu):
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
        self.net = caffe.Net(model_def,
                             model_weights,
                             caffe.TEST)

        N, C, H, W = list(self.net.blobs['data'].shape)
        self.transformer = caffe.io.Transformer({'data': (1, 3, H, W)})
        self.net.blobs['data'].reshape(1, 3, H, W)
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104, 117, 123]))
        self.transformer.set_raw_scale('data', 255)
        self.transformer.set_channel_swap('data', (2, 1, 0))

    def detect(self, image_file, conf_thresh=0.5, topn=200):
        image = caffe.io.load_image(image_file)
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        start = time.time()
        detections = self.net.forward()['detection_out']
        end = time.time()
        diff = end - start
        # Note: this is not how the models were benchmarked in the paper, the first forward pass takes a lot more time than consecutive ones
        print("Forward pass + NMS took {:.3f} ({:.2f} FPS).".format(diff, 1. / diff))

        # Parse the outputs.
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]

        # Get detections with confidence higher than conf_thresh
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        result = []
        for i in range(min(topn, top_conf.shape[0])):
            xmin = top_xmin[i]
            ymin = top_ymin[i]
            xmax = top_xmax[i]
            ymax = top_ymax[i]
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = 'Object'
            result.append([xmin, ymin, xmax, ymax, label, score, label_name])
        return result


def main(args):
    model = glob.glob(osp.join(args.model_path, '*.prototxt'))[0]
    weights = glob.glob(osp.join(args.model_path, '*.caffemodel'))[0]

    detection = CaffeDetection(model, weights, args.gpu)

    result = detection.detect(args.image,
                              conf_thresh=args.conf_thresh,
                              )
    frame = cv2.imread(args.image)
    height, width, _ = frame.shape
    print("{} detections.".format(len(result)))

    for item in result:
        xmin = int(round(item[0] * width))
        ymin = int(round(item[1] * height))
        xmax = int(round(item[2] * width))
        ymax = int(round(item[3] * height))
        conf = round(item[-2] * 100, 1)

        # Color
        color_border = (255, 255, 255)
        cv2.rectangle(frame, (xmin + 1, ymax - 1), (xmax - 1, ymin + 1), (0, 0, 0), thickness=2)
        cv2.rectangle(frame, (xmin, ymax), (xmax, ymin), color_border, thickness=2)
        cv2.rectangle(frame, (xmin - 1, ymax + 1), (xmax + 1, ymin - 1), (0, 0, 0), thickness=1)

    out_path = osp.abspath(osp.join('.', 'detect_result.jpg'))
    cv2.imwrite(out_path, frame)
    print("Result written to {}.".format(out_path))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--conf-thresh', default=0.5, type=float)
    parser.add_argument('--image', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
