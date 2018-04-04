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
flags.DEFINE_integer("batch_size", 32, "The number of batch images [32]")
flags.DEFINE_integer("image_size", 148, "The size of image to use (will be center cropped) [148]")
flags.DEFINE_integer("output_size", 128, "The size of the output images to produce [128]")
flags.DEFINE_integer("sample_size", 128, "The number of sample images [128]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("z_dim", 100, "Dimension of latent representation vector from. [100]")

flags.DEFINE_string("input","", "Provide the name of the model to freeze [-]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(FLAGS.__flags)

    # input image
    input_imgs = tf.placeholder(tf.float32,[FLAGS.batch_size, FLAGS.output_size, 
            FLAGS.output_size, FLAGS.c_dim], name='input_image')

    # ----------------------encoder----------------------
    net_out1, net_out2, z_mean, z_log_sigma_sq = encoder(input_imgs, is_train=False, reuse=False)
    read_z = tf.identity(z_mean, name='read_z')

    # ----------------------decoder----------------------
    gen0, gen0_logits = generator(z_mean, is_train=False, reuse=False) # reconstruction
    read_gen0 = tf.identity(gen0.outputs, name='output_image')
    
    # create session
    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)
    
    # load checkpoint params
    print '\nLoading model: '+str(FLAGS.input)
    load_params = tl.files.load_npz(name=FLAGS.input+'_e1.npz')
    tl.files.assign_params(sess, load_params, net_out1)
    load_params = tl.files.load_npz(name=FLAGS.input+'_e2.npz')
    tl.files.assign_params(sess, load_params, net_out2)
    load_params = tl.files.load_npz(name=FLAGS.input+'_g.npz')
    tl.files.assign_params(sess, load_params, gen0)

    # freeze and save graph
    freezedFile = os.path.splitext(FLAGS.input)[0]+'_frz.pb'
    print 'Creating freezed graph: '+freezedFile
    graph_frz = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['input_image','read_z','output_image'])
    tf.train.write_graph(graph_frz, '.', freezedFile, as_text=False)

    # save as ckpt for the web
    saver = tf.train.Saver()
    checkpoint_file = os.path.join(os.path.splitext(FLAGS.input)[0]+'_web', 'model.ckpt')
    saver.save(sess, checkpoint_file)


    # create and save sample image
    #img1 = sess.run(gen0.outputs, feed_dict={})
    #save_images(img1, [8, 8],'./freeze-debug.png')
    #print 'Generated image: ./freeze-debug.png'


if __name__ == '__main__':
    tf.app.run()

