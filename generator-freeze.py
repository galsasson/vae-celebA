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

flags.DEFINE_string("input","", "Provide the name of the model to freeze [-]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(FLAGS.__flags)

    # normal distribution for generator
    print '\nCreative generator model...'
    z_p = tf.random_normal(shape=(FLAGS.batch_size, FLAGS.z_dim), mean=0.0, stddev=1.0, name='z_input')
    
    # ----------------------decoder----------------------
    gen0, gen0_logits = generator(z_p, is_train=False, reuse=False) # reconstruction
    
    # create session
    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)
    
    # load checkpoint params
    print '\nLoading model: '+str(FLAGS.input)
    load_params = tl.files.load_npz(name=FLAGS.input)
    tl.files.assign_params(sess, load_params, gen0)


    img1 = sess.run(gen0.outputs, feed_dict={})


    freezedFile = os.path.splitext(FLAGS.input)[0]+'_frz.meta'
    print '\n Creating freezed graph: '+freezedFile
    tf.train.export_meta_graph(freezedFile, as_text=True)

    # save image
    save_images(img1, [8, 8],'./debug.png')
    print 'Generated image: ./debug.png'


if __name__ == '__main__':
    tf.app.run()

