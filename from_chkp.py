#!/usr/bin/env python3
# -*- coding:utf8 -*-
import argparse
import tensorflow as tf


def main(chkp_path):
    print("Loading {}".format(chkp_path))
    meta_graph_path = "{}.meta".format(chkp_path)
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        saver = tf.train.import_meta_graph(meta_graph_path, clear_devices=True)
        saver.restore(sess, chkp_path)
    for node in graph.as_graph_def().node:
        print(node.name, node.op)
    
    writer = tf.summary.FileWriter(logdir="graph_logs/restore_graph",
                                   graph=graph)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chkp-path", dest="chkp_path",
                        default="checkpoints", metavar="CHKP_PREFIX",
                        help="the checkpoint path (default: checkpoints)")
    args = vars(parser.parse_known_args()[0])
    main(**args)
