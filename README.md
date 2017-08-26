## Dive into Freeze Graph

This is a repo for my blog post.

In that post, I dive into the source code of Google `Tensorflow` (the python part :p) to 

show you how `Tensorflow` internally integrated with `Protobuf`, a tool developed by Googel

for data serailization supporting mulitple language (it's really cool, you should try it!)

### Example Code

I used to use VGG16/19 as example since their architectures seem easier to understand for lazy

me, but I've heard that AlexNet is very fast and can run almost real-time on a mobile device

(I havn't tried it though). So, I'll use AlexNet as example in this repo.

I borrow some code by **kratzert** (the [repo](https://github.com/kratzert/finetune_alexnet_with_tensorflow/tree/5d751d62eb4d7149f4e3fd465febf8f07d4cea9d)). Thanks a lot! Nicely organized `Tensorflow` project.

You can find pretrained weights (.npy) [here](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy). **Make sure you have it before running any code**.

The label txt file is found [here](https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57)
