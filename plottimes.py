#!/home/gal/google/vae-celebA/env2/bin/python
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

flags = tf.app.flags
flags.DEFINE_string("file", "", "Provide times file to load []")
flags.DEFINE_string("plot", "all", "Provide a list of value names to plot (vae,kl,sse,p1,p2,p3,p) [all]")
FLAGS = flags.FLAGS

def usage():
	print 'Usage:'
	print ' --file			Provide times file to load []'
	print ' --plot 			Provide a list of value names to plot (vae,kl,sse,p1,p2,p3,p) [all]'
	return

if FLAGS.file == '':
	usage()
	exit()

names=['vae','kl','sse','p1','p2','p3','p']
values=[[] for y in range(len(names))]
steps=[]

with open(FLAGS.file,'r') as f:
	lines = f.readlines()
	for line in lines:
		numbers = line.rstrip().split(',')
		steps.append(int(numbers[0]))
		for i in range(len(names)):
			values[i].append(float(numbers[i+1]))


toplot = FLAGS.plot.split(',')

for p in range(len(names)):
	if FLAGS.plot == 'all' or any(names[p] in s for s in toplot):
		fig = plt.figure(p+1)
		sp = fig.add_subplot(1,1,1)
		plt.plot(steps,values[p])
		plt.locator_params(nbins=10)
		plt.ylabel(names[p] + ' loss')
		plt.title(names[p] + ' loss ('+FLAGS.file+')')


plt.show()
