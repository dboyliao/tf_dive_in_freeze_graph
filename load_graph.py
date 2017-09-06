#!/usr/bin/env python3
# -*- coding:utf8 -*-
import argparse
import tensorflow as tf


def main(pb_file):
    print("Loading {}".format(pb_file))
    graph_def = tf.GraphDef()
    with open(pb_file, "rb") as fid:
        graph_def.ParseFromString(fid.read())
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")

    for node in graph.as_graph_def().node:
        print(node.name, node.op)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pb_file", metavar="FILE.pb",
                        nargs='?',
                        default="graph.pb",
                        help="model protobuf file to load (default: graph.pb)")
    args = vars(parser.parse_known_args()[0])
    main(**args)