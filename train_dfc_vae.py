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
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

'''
Tensorlayer implementation of DFC-VAE
'''

flags = tf.app.flags
flags.DEFINE_integer("epoch", 50, "Epoch to train [50]") 
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam [0.001]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 32, "The number of batch images [32]")
flags.DEFINE_integer("image_size", 148, "The size of image to use (will be center cropped) [148]")
flags.DEFINE_integer("output_size", 128, "The size of the output images to produce [128]")
flags.DEFINE_integer("sample_size", 128, "The number of sample images [128]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("z_dim", 100, "Dimension of latent representation vector from. [100]")
flags.DEFINE_integer("sample_step", 500, "The interval of generating sample. [500]")
flags.DEFINE_integer("save_step", 500, "The interval of saveing checkpoints. [500]")
flags.DEFINE_string("dataset", "img_align_celeba", "The name of dataset [img_align_celeba]")
flags.DEFINE_string("test_name", "testname", "The number of experiment [testname]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("times_dir", "times", "Directory name to save times [times]")
flags.DEFINE_boolean("random_crop", False, "True to perform random cropping (non centered) [False]")
flags.DEFINE_string("load_model","no", "Provide the name of the model to load [no]")
flags.DEFINE_integer("init_blur", 0, "Initial training on blurred images. [0]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(FLAGS.__flags)

    # prepare for the file directory
    tl.files.exists_or_mkdir(FLAGS.checkpoint_dir)
    tl.files.exists_or_mkdir(FLAGS.sample_dir)

    with tf.device("/gpu:0"):
        ##========================= DEFINE MODEL ===========================##
        # the input_imgs are input for both encoder and discriminator
        input_imgs = tf.placeholder(tf.float32,[FLAGS.batch_size, FLAGS.output_size, 
            FLAGS.output_size, FLAGS.c_dim], name='real_images')

        # normal distribution for generator
        z_p = tf.random_normal(shape=(FLAGS.batch_size, FLAGS.z_dim), mean=0.0, stddev=1.0)
        # normal distribution for reparameterization trick
        eps = tf.random_normal(shape=(FLAGS.batch_size, FLAGS.z_dim), mean=0.0, stddev=1.0)
        lr_vae = tf.placeholder(tf.float32, shape=[])


        # ----------------------encoder----------------------
        net_out1, net_out2, z_mean, z_log_sigma_sq = encoder(input_imgs, is_train=True, reuse=False)

        # ----------------------decoder----------------------
        # decode z 
        # z = z_mean + z_sigma * eps
        z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps)) # using reparameterization tricks
        gen0, gen0_logits = generator(z, is_train=True, reuse=False) # reconstruction

        # ----------------------vgg net--------------------------
        vgg1_input = tf.image.resize_images(input_imgs,[224,224])
        net_in_real = InputLayer(vgg1_input, name='input1')
        conv1,l1_r,l2_r,l3_r,_,_ = conv_layers_simple_api(net_in_real,reuse=False)
        vgg1 = fc_layers(conv1,reuse=False)

        vgg2_input = tf.image.resize_images(gen0.outputs,[224,224])
        net_in_fake = InputLayer(vgg2_input, name='input2')
        conv2,l1,l2,l3,_,_ = conv_layers_simple_api(net_in_fake,reuse=True)
        vgg2 = fc_layers(conv2,reuse=True)


        # ----------------------for samples----------------------
        gen2, gen2_logits = generator(z, is_train=False, reuse=True)
        gen3, gen3_logits = generator(z_p, is_train=False, reuse=True)

        ##========================= DEFINE TRAIN OPS =======================##
        ''''
        reconstruction loss:
        use the learned similarity measurement in l-th layer(feature space) of pretrained VGG-16
        '''

        SSE_loss = tf.reduce_mean(tf.reduce_sum(tf.square(gen0.outputs - input_imgs),[1,2,3]))
        print(SSE_loss.get_shape(),type(SSE_loss))

        # perceptual loss in feature space in VGG net
        p1_loss = tf.reduce_mean(tf.reduce_sum(tf.square(l1 - l1_r), [1,2,3]))
        p2_loss = tf.reduce_mean(tf.reduce_sum(tf.square(l2 - l2_r), [1,2,3]))
        p3_loss = tf.reduce_mean(tf.reduce_sum(tf.square(l3 - l3_r), [1,2,3]))
        p_loss = p1_loss + p2_loss + p3_loss
        '''
        KL divergence:
        we get z_mean,z_log_sigma_sq from encoder, then we get z from N(z_mean,z_sigma^2)
        then compute KL divergence between z and standard normal gaussian N(0,I) 
        '''
        # train_vae
        KL_loss = tf.reduce_mean(- 0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq),1))
        print(KL_loss.get_shape(),type(KL_loss))

        ### important points! ###
        style_content_weight = 3e-5 # you may need to tweak this weight for a different dataset
        VAE_loss = KL_loss + style_content_weight*p_loss

        e_vars = tl.layers.get_variables_with_name('encoder',True,True)
        g_vars = tl.layers.get_variables_with_name('generator', True, True)
        vae_vars = e_vars + g_vars

        print("-------encoder-------")
        net_out1.print_params(False)
        print("-------generator-------")
        gen0.print_params(False)


        # optimizers for updating encoder and generator
        vae_optim = tf.train.AdamOptimizer(lr_vae, beta1=FLAGS.beta1) \
                           .minimize(VAE_loss, var_list=vae_vars)
    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)

    npz = np.load('vgg16_weights.npz')
    params = []
    for val in sorted( npz.items() ):
        print("  Loading %s" % str(val[1].shape))
        params.append(val[1])
    tl.files.assign_params(sess, params, vgg1)
    tl.files.assign_params(sess, params, vgg2)


    # load checkpoint params
    if FLAGS.load_model != "no":
        print 'Loading model: '+str(FLAGS.load_model)
        load_params = tl.files.load_npz(name=FLAGS.load_model+'_e1.npz')
        tl.files.assign_params(sess, load_params, net_out1)
        load_params = tl.files.load_npz(name=FLAGS.load_model+'_e2.npz')
        tl.files.assign_params(sess, load_params, net_out2)
        load_params = tl.files.load_npz(name=FLAGS.load_model+'_g.npz')
        tl.files.assign_params(sess, load_params, gen0)

    # create checkpoint dir
    save_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.test_name) #'./checkpoint/vae_0808'
    tl.files.exists_or_mkdir(save_dir)
    # create samples dir
    samples_dir = FLAGS.sample_dir + "/" + FLAGS.test_name
    tl.files.exists_or_mkdir(samples_dir)
    # create times dir and file
    tl.files.exists_or_mkdir(FLAGS.times_dir)
    timesFilename = FLAGS.times_dir + "/" + FLAGS.test_name + ".times"
    with open(timesFilename, "w") as f:
        f.write("");    # clean file

    # get the list of absolute paths of all images in dataset
    data_files = glob(os.path.join("./data", FLAGS.dataset, "*.jpg"))
    data_files = sorted(data_files)
    data_files = np.array(data_files) # for tl.iterate.minibatches


    ##========================= TRAIN MODELS ================================##
    iter_counter = 0

    training_start_time = time.time()

    blurVal = FLAGS.init_blur

    # use all images in dataset in every epoch
    for epoch in range(FLAGS.epoch):
        ## shuffle data
        print("[*] Dataset shuffled!")

        minibatch = tl.iterate.minibatches(inputs=data_files, targets=data_files, batch_size=FLAGS.batch_size, shuffle=True)
        idx = 0
        batch_idxs = min(len(data_files), FLAGS.train_size) // FLAGS.batch_size
        blurVal -= 6
        if blurVal < 0:
            blurVal = 0

        while True:
            try:
                batch_files,_ = minibatch.next()
                batch = [get_image(batch_file, FLAGS.image_size, is_crop=True, resize_w=FLAGS.output_size, is_grayscale = 0, blur=blurVal, is_centered=not FLAGS.random_crop) \
                        for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                start_time = time.time()
                vae_current_lr = FLAGS.learning_rate


                # update
                p, p1, p2, p3, kl, sse, errE, _ = sess.run([p_loss,p1_loss,p2_loss,p3_loss,KL_loss,SSE_loss,VAE_loss,vae_optim], feed_dict={input_imgs: batch_images, lr_vae:vae_current_lr})


                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, vae_loss:%.2f, kl_loss:%.2f, sse_loss:%.2f, p1_loss:%.2f, p2_loss:%.2f, p3_loss:%.2f, p_loss:%.2f" \
                        % (epoch, FLAGS.epoch, idx, batch_idxs,
                            time.time() - start_time, errE, kl, sse, p1, p2, p3, p))
                sys.stdout.flush()

                iter_counter += 1
                # save samples
                if np.mod(iter_counter, FLAGS.sample_step) == 0:

                    # generate and visualize generated images
                    img1, img2 = sess.run([gen2.outputs, gen3.outputs], feed_dict={input_imgs: batch_images})
                    save_images(img1, [8, 8],
                                './{}/train_{:d}.png'.format(samples_dir, iter_counter))

                    # img2 = sess.run(gen3.outputs, feed_dict={input_imgs: batch_images})
                    save_images(img2, [8, 8],
                                './{}/train_{:d}_random.png'.format(samples_dir, iter_counter))

                    # save input image for comparison
                    save_images(batch_images,[8, 8],'./{}/train_{:d}_input.png'.format(samples_dir, iter_counter))
                    print("[Sample] sample generated!!!")
                    sys.stdout.flush()

                    # write times to file
                    with open(timesFilename, "a") as file:
                        file.write("%8d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n" % (iter_counter, errE, kl, sse, p1, p2, p3, p))

                # save checkpoint
                if np.mod(iter_counter, FLAGS.save_step) == 0:
                    # save current network parameters
                    print("[*] Saving checkpoints...")
                    net_e1_name = os.path.join(save_dir, 'net_%d_e1.npz' % iter_counter)
                    net_e2_name = os.path.join(save_dir, 'net_%d_e2.npz' % iter_counter)
                    net_g_name = os.path.join(save_dir, 'net_%d_g.npz' % iter_counter)

                    tl.files.save_npz(net_out1.all_params, name=net_e1_name, sess=sess)
                    tl.files.save_npz(net_out2.all_params, name=net_e2_name, sess=sess)
                    tl.files.save_npz(gen0.all_params, name=net_g_name, sess=sess)

                    print("[*] Saving checkpoints SUCCESS!")

                idx += 1
                # print idx
            except StopIteration:
                print 'one epoch finished'
                break

            


    training_end_time = time.time()
    print("The processing time of program is : {:.2f}mins".format((training_end_time-training_start_time)/60.0))


if __name__ == '__main__':
    tf.app.run()

