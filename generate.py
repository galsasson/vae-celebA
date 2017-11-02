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
flags.DEFINE_integer("z_dim", 100, "Dimension of latent representation vector from. [2048]")

flags.DEFINE_string("model","", "Provide the name of the model to load [-]")
flags.DEFINE_string("input", "random", "Provide input file with z to generate [random]")
flags.DEFINE_string("output", "image", "Output image name [image]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(FLAGS.__flags)

    with tf.device("/gpu:0"):
        ##========================= DEFINE MODEL ===========================##
        # the input_imgs are input for both encoder and discriminator
        input_imgs = tf.placeholder(tf.float32,[FLAGS.batch_size, FLAGS.output_size, 
            FLAGS.output_size, FLAGS.c_dim], name='real_images')

        # normal distribution for generator
        z_p = tf.random_normal(shape=(FLAGS.batch_size, FLAGS.z_dim), mean=0.0, stddev=1.0)

        # ----------------------decoder----------------------
        # decode z 
        gen0, gen0_logits = generator(z_p, is_train=False, reuse=False) # reconstruction

        sess = tf.InteractiveSession()
        tl.layers.initialize_global_variables(sess)

    # load checkpoint params
    print
    print 'Loading model: '+str(FLAGS.model)
    load_params = tl.files.load_npz(name=FLAGS.model+'_g.npz')
    tl.files.assign_params(sess, load_params, gen0)

    # generate and visualize generated images
    if FLAGS.input == "random":
        print 'Generating random samples...'
        img1 = sess.run(gen0.outputs, feed_dict={})
        save_images(img1, [8, 8],'./'+FLAGS.output+'.png')
        print 'generated image: ./'+FLAGS.output+'.png'


if __name__ == '__main__':
    tf.app.run()

