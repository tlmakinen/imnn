import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from IMNN import IMNN
from IMNN.LFI import LFI
from make_data import GenerateGaussianNoise
from make_data import AnalyticLikelihood
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

import sys,os

tfd = tfp.distributions

print("IMNN {}\nTensorFlow {}\nTensorFlow Probability {}\nnumpy {}".format(
    IMNN.__version__, tf.__version__, tfp.__version__, np.__version__))


from make_2Dfield import GenerateCosmoField, GenerateCosmoFieldOneParam

from nn_model import field2Dmodel

def fiducial_loader(seed, data):
    yield data[seed], seed

def derivative_loader(seed, derivative, parameter, data):
    yield data[seed, derivative, parameter], (seed, derivative, parameter)



# create several test run functions for testing the three
# inference scenarios
# SCENARIO 1: vanilla, ONE parameter, A=2.0; NO foreground
# SCENARIO 2: foreground, ONE paramter, θ_fid = A_fid = 2.0, θ_fg = [3.8, 2.0]

# SCENARIO 3: foreground, TWO paramers, θ_fid = [2.0, 0.8], θ_fg = [3.8, 2.0]

def run_vanilla_one_nofg(filename, θ_fid = np.array([2.0]), n_sims=10000):

	####### define fiducial model #######
	θ_fid = θ_fid # scale, power=0.5

	print('running test IMNN inference on ')


	SN = GenerateCosmoFieldOneParam(n_s=10, n_d=10, n_params=1, n_summaries=1, 
				input_shape=(1,128,128), θ_fid=θ_fid, θ_fg=None)

	# FOR NOW: load data in-memory (until Tom finishes fast directory loader)
	print("now I'm generating the simulations ...")
	details, fiducial, validation_fiducial, derivative, validation_derivative = SN.generate_data()  

	print("fiducial = {}\nvalidation_fiducial = {}\nderivative = {}\nvalidation_derivative = {}".format(
	    fiducial.shape, validation_fiducial.shape, derivative.shape, validation_derivative.shape))

	model = field2Dmodel(SN.input_shape, SN.n_summaries, kernel=3)

	opt = tf.keras.optimizers.Adam()

	# set up IMNN

	imnn = IMNN.IMNN(n_s=SN.n_s, n_d=SN.n_d, n_params=SN.n_params, n_summaries=SN.n_summaries,
                 model=model, optimiser=opt, θ_fid=θ_fid, δθ=np.array([0.1]), input_shape=SN.input_shape,
                 fiducial=lambda x : fiducial_loader(x, fiducial), 
                 derivative=lambda x, y, z : derivative_loader(x, y, z, derivative), 
                 validation_fiducial=lambda x : fiducial_loader(x, validation_fiducial),
                 validation_derivative=lambda x, y, z : derivative_loader(x, y, z, validation_derivative), 
                 at_once=SN.n_s, check_shape=True, verbose=True,
                 directory="model", filename=filename, save=True, load=True, weights='weights')


	imnn.fit(patience=10, min_iterations=1000, tqdm_notebook=True, checkpoint=100)

	# do population monte carlo sampling using trained model


def run_vanilla_one_fg(filename, n_sims=10000):

	####### define fiducial model #######
	θ_fid = np.array([2.0]) # scale, power=0.5
	θ_fg = np.array([1.8, 3.2])

	print('running test IMNN training on amp + fg \nθ_fid ={}, θ_fg={}'.format(θ_fid, θ_fg))

	
	SN = GenerateCosmoFieldOneParam(n_s=10, n_d=10, n_params=1, n_summaries=1, 
				input_shape=(1,128,128), θ_fid=θ_fid, θ_fg=θ_fg)


	# FOR NOW: load data in-memory (until Tom finishes fast directory loader)
	print("now I'm generating the simulations ...")
	details, fiducial, validation_fiducial, derivative, validation_derivative = SN.generate_data()  

	print("fiducial = {}\nvalidation_fiducial = {}\nderivative = {}\nvalidation_derivative = {}".format(
	    fiducial.shape, validation_fiducial.shape, derivative.shape, validation_derivative.shape))

	model = field2Dmodel(SN.input_shape, SN.n_summaries, kernel=3)

	opt = tf.keras.optimizers.Adam()

	# set up IMNN

	imnn = IMNN.IMNN(n_s=SN.n_s, n_d=SN.n_d, n_params=SN.n_params, n_summaries=SN.n_summaries,
                 model=model, optimiser=opt, θ_fid=θ_fid, δθ=np.array([0.1]), input_shape=SN.input_shape,
                 fiducial=lambda x : fiducial_loader(x, fiducial), 
                 derivative=lambda x, y, z : derivative_loader(x, y, z, derivative), 
                 validation_fiducial=lambda x : fiducial_loader(x, validation_fiducial),
                 validation_derivative=lambda x, y, z : derivative_loader(x, y, z, validation_derivative), 
                 at_once=SN.n_s, check_shape=True, verbose=True,
                 directory="model", filename=filename, save=True, load=True, weights='weights')


	imnn.fit(patience=10, min_iterations=1000, tqdm_notebook=False, checkpoint=100)	


def run_cosmo_fg(filename, θ_fid = np.array([2.0, 0.8]), θ_fg = np.array([1.8, 3.2]), n_sims=10000):

	####### define fiducial model #######
	θ_fid = θ_fid # scale, power=0.5
	θ_fg = θ_fg



	print('running test IMNN training on cosmo + fg \nθ_fid ={}, θ_fg={}'.format(θ_fid, θ_fg))
	print('saving to model directory %s'%(filename))
	
	SN = GenerateCosmoFieldOneParam(n_s=n_sims, n_d=n_sims, input_shape=(1,128,128), θ_fid=θ_fid, θ_fg=θ_fg)


	# FOR NOW: load data in-memory (until Tom finishes fast directory loader)
	print("now I'm generating the simulations ...")
	details, fiducial, validation_fiducial, derivative, validation_derivative = SN.generate_data()  

	print("fiducial = {}\nvalidation_fiducial = {}\nderivative = {}\nvalidation_derivative = {}".format(
	    fiducial.shape, validation_fiducial.shape, derivative.shape, validation_derivative.shape))

	model = field2Dmodel(SN.input_shape, SN.n_summaries, kernel=3)

	opt = tf.keras.optimizers.Adam()

	# set up IMNN

	imnn = IMNN.IMNN(n_s=SN.n_s, n_d=SN.n_d, n_params=SN.n_params, n_summaries=SN.n_summaries,
                 model=model, optimiser=opt, θ_fid=θ_fid, δθ=np.array([0.1, 0.1]), input_shape=SN.input_shape,
                 fiducial=lambda x : fiducial_loader(x, fiducial), 
                 derivative=lambda x, y, z : derivative_loader(x, y, z, derivative), 
                 validation_fiducial=lambda x : fiducial_loader(x, validation_fiducial),
                 validation_derivative=lambda x, y, z : derivative_loader(x, y, z, validation_derivative), 
                 at_once=SN.n_s, check_shape=True, verbose=True,
                 directory="model", filename=filename, save=True, load=False, weights='weights')


	imnn.fit(patience=10, min_iterations=1000, tqdm_notebook=False, checkpoint=100)	






if __name__ == '__main__':

	# run all tests according to slurm index
	job = int(sys.argv[1])

	if job == 1:
		run_vanilla_one_nofg('amp-vanilla', n_sims=10)

	if job == 2:
		run_vanilla_one_fg('amp+fg', n_sims=10)

	if job == 3:
		run_cosmo_fg('cosmo+fg', n_sims=10)

	if job == 4:
		run_cosmo_fg('cosmo+big_fg', θ_fg = np.array([100.8, 3.2]), n_sims=10)









