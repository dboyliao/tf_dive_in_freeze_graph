#!/usr/bin/env python3
# -*- coding:utf8 -*-
import argparse
import tensorflow as tf
from build_alexnet import AlexNet


def main(chkp_path, npy_path=None, labels_file=None):
    """
    main function
    """
    # pylint: disable=C0301
    _ = print("Building with pretrained weights: {}".format(npy_path)) if npy_path else None
    _ = print("Building with labels file: {}".format(labels_file)) if labels_file else None
    # pylint: enable=C0301

    net = AlexNet(npy_path=npy_path, labels_file=labels_file)
    with tf.Session(graph=net._graph) as sess:
        saver = tf.train.Saver()
        tf.global_variables_initializer().run()
        saver.save(sess, save_path=chkp_path)
    print("Saving session with save path {}".format(chkp_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-npy", dest="npy_path",
                        metavar="WEIGHTS.npy",
                        help="input npy file (default: None)",
                        default=None)
    parser.add_argument("-l", "--labels-file", dest="labels_file",
                        metavar="LABELS.txt",
                        help="labels file (default: None)",
                        default=None)
    parser.add_argument("--chkp-path", dest="chkp_path",
                        metavar="CHKP_PREFIX",
                        help="checkpoint path (default: save_chkp)",
                        default="save_chkp")
    args = vars(parser.parse_known_args()[0])
    main(**args)
