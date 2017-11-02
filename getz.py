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
flags.DEFINE_string("input", "", "Provide input image [-]")
flags.DEFINE_string("output", "zvec", "Output file name [zvec]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(FLAGS.__flags)

    with tf.device("/gpu:0"):
        ##========================= DEFINE MODEL ===========================##
        # the input_imgs are input for both encoder and discriminator
        input_imgs = tf.placeholder(tf.float32,[FLAGS.batch_size, FLAGS.output_size, 
            FLAGS.output_size, FLAGS.c_dim], name='real_images')

        # normal distribution for reparameterization trick
        eps = tf.random_normal(shape=(FLAGS.batch_size, FLAGS.z_dim), mean=0.0, stddev=1.0)

        # ----------------------encoder----------------------
        net_out1, net_out2, z_mean, z_log_sigma_sq = encoder(input_imgs, is_train=False, reuse=False)

        # ----------------------decoder----------------------
        # decode z 
        # z = z_mean + z_sigma * eps
        z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps)) # using reparameterization tricks

    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)

    # load checkpoint params
    print
    print 'Loading model: '+str(FLAGS.model)
    load_params = tl.files.load_npz(name=FLAGS.model+'_e1.npz')
    tl.files.assign_params(sess, load_params, net_out1)
    load_params = tl.files.load_npz(name=FLAGS.model+'_e2.npz')
    tl.files.assign_params(sess, load_params, net_out2)

    # images to reconstruct
    image_files = FLAGS.input.split(',')
    while len(image_files)<FLAGS.batch_size:
        image_files.append("black.jpg");

    batch = [get_image(batch_file, FLAGS.image_size, is_crop=True, resize_w=FLAGS.output_size, is_grayscale = 0) for batch_file in image_files]
    batch_images = np.array(batch).astype(np.float32)

    save_images(batch_images,[8, 8],'./'+FLAGS.output+'_input.png')
    print 'input image: ./'+FLAGS.output+'_input.png'

    # generate and visualize generated images
    tmpz = sess.run(z, feed_dict={input_imgs: batch_images})
    with open(FLAGS.output+'.txt', 'w') as f:
        f.write(tmpz)
    print 'output file: ./'+FLAGS.output+'.txt'
#    save_images(img1, [8, 8],'./'+FLAGS.output+'_z.png')
#    print 'reconstructed image: ./'+FLAGS.output+'_reconstruct.png'


if __name__ == '__main__':
    tf.app.run()

