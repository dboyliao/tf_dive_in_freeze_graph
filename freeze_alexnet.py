#!/usr/bin/env python3
# -*- coding: utf8 -*-
import argparse
from build_alexnet import AlexNet
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants


def main(out_fname="alexnet.pb", npy_path="bvlc_alexnet.npy"):
    net = AlexNet(npy_path)
    with tf.Session(graph=net._graph) as sess:
        tf.global_variables_initializer().run()
        freeze_graph_def =  convert_variables_to_constants(sess,
                                                           net._graph.as_graph_def(),
                                                           [net.logits.op.name])
        with tf.gfile.GFile(out_fname, "wb") as fid:
            fid.write(freeze_graph_def.SerializeToString())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out-fname", dest="out_fname",
                        help="output .pb file name", metavar="FILE.pb",
                        default="alexnet.pb")
    parser.add_argument("-i", "--npy-fname", dest="npy_path",
                        help="npy file path", metavar="FILE.npy",
                        default="bvlc_alexnet.npy")
    args = vars(parser.parse_known_args()[0])
    main(**args)

