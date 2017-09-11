#!/usr/bin/env python3
# -*- coding: utf8 -*-
import argparse
from build_alexnet import AlexNet
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants


def main(npy_path="bvlc_alexnet.npy",
         label_path="imagenet_labels.txt",
         out_fname="alexnet.pb"):
    print("npy file: {}".format(npy_path))
    print("label file: {}".format(label_path))
    net = AlexNet(npy_path, label_path)
    with tf.Session(graph=net._graph) as sess:
        tf.global_variables_initializer().run()
        freeze_graph_def = convert_variables_to_constants(sess,
                                                          net._graph.as_graph_def(),
                                                          [net.logits.op.name])
        with tf.gfile.GFile(out_fname, "wb") as fid:
            fid.write(freeze_graph_def.SerializeToString())
    print("output file: {}".format(out_fname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out-fname", dest="out_fname",
                        help="output .pb file name", metavar="FILE.pb",
                        default="alexnet.pb")
    parser.add_argument("npy_path", nargs='?',
                        help="npy file path", metavar="FILE.npy",
                        default="bvlc_alexnet.npy")
    parser.add_argument("label_path", nargs="?",
                        help="imagenet label file path", metavar='LABELS.txt',
                        default="imagenet_labels.txt")
    args = vars(parser.parse_known_args()[0])
    main(**args)

