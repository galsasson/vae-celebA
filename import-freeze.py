import os
import sys
import scipy.misc
import pprint
import numpy as np
import time
import math
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from glob import glob
from random import shuffle
from dfc_vae import *
from utils import *
from vgg_loss import *

pp = pprint.PrettyPrinter()

'''
Tensorlayer implementation of DFC-VAE
'''

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 32, "The number of batch images [64]")
flags.DEFINE_integer("image_size", 148, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 128, "The size of the output images to produce [64]")
flags.DEFINE_integer("sample_size", 128, "The number of sample images [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("z_dim", 100, "Dimension of latent representation vector from. [100]")

flags.DEFINE_string("model","", "Provide the name of the model to load [-]")
flags.DEFINE_string("input", "random", "Provide input file with z to generate [random]")
flags.DEFINE_string("output", "image", "Output image name [image]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(FLAGS.__flags)

    # generate and visualize generated images
    print '\nGenerating from file: '+FLAGS.input

    # read z from file
    z = np.load(FLAGS.input)

    # read freezed graph
    with tf.gfile.GFile(FLAGS.model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        
    # create session
    tf.import_graph_def(graph_def)
    sess = tf.Session()
    tl.layers.initialize_global_variables(sess)

    # run computation
    img1 = sess.run('import/generator/Tanh:0', feed_dict={'import/z_input:0':z})

    # save image
    save_images(img1, [8, 8],'./'+FLAGS.output+'.png')
    print 'Generated image: ./'+FLAGS.output+'.png'

if __name__ == '__main__':
    tf.app.run()

