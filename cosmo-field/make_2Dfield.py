import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IMNN.utils import TFRecords
from IMNN.LFI.LFI import GaussianApproximation

from FyeldGenerator import generate_field


__version__ = "0.2a5"
__author__ = "Lucas Makinen"

# HERE θ_fid is [A,B], yielding power spectrum P(k) = A k^-B

class GenerateCosmoField():
    def __init__(self, input_shape=(1,128,128), n_params=2, n_summaries=2, n_s=1000, n_d=1000, n_d_small=100,
                 θ_fid=np.array([1.0, 0.5]), δθ=np.array([0.2, 0.1]), θ_fg=None, training_seed=0,
                 validation_seed=1):
        
        self.input_shape = input_shape
        self.n_params = n_params
        self.n_summaries = n_summaries
        self.n_s = n_s
        self.n_d = n_d
        self.n_d_small = n_d_small
        self.θ_fid = θ_fid
        self.δθ = δθ
        self.half_δθ = δθ / 2.
        self.training_seed = training_seed
        self.validation_seed = validation_seed
        self.θ_fg = θ_fg

    def get_fiducial(self, seed, data):
        return data[seed]

    def get_derivative(self, seed, derivative, parameter, data):
        return data[seed, derivative, parameter]

    def check_selection(self, size):
        if size not in ["full", "all", "small"]:
            print("size must be `full`, `all` or `small` describing, respectively "
                  "whether just `n_d=n_s` is returned, or `n_d=n_s` and `n_d_small` "
                  "is returned, or `n_d=n_d_small` is returned.")
            sys.exit()
    
    def check_ftype(self, ftype):
        if ftype not in ["both", "numpy", "tfrecords"]:
            print("size must be `both`, `numpy` or `tfrecords` describing, respectively "
                  "whether both `numpy` and `tfrecords` files are saved, or just either one.")
            sys.exit()

    # Helper that generates power-law power spectrum
    def Pkgen(self, n, amp=1):
        def Pk(k):
            return amp*np.power(k, -n)

        return Pk

    # Draw samples from a normal distribution
    def distrib(self, shape):
        a = np.random.normal(loc=0, scale=1, size=shape)
        b = np.random.normal(loc=0, scale=1, size=shape)
        return a + 1j * b

    def simulator(self, parameters, θ_fg=None, seed=None, save_fg_copy=False,
                  fg_repeat=False,
                  simulator_args=None):
        # if self.input_shape[0] // 3 != 0:
        #     raise AssertionError ("input shape must be divisible by coordinate dimensions !")
        if seed is not None:
            np.random.seed(seed)
        if len(parameters.shape) == 1:
            parameters = parameters[np.newaxis, :]
        # if only looking at amplitude, fix power to 0.5
        if self.n_params == 1:
            parameters = np.repeat(parameters, 2, axis=1)
            parameters[:, 1] = np.ones_like(parameters[:, 1])*0.5

        
        d = np.array([generate_field(self.distrib, self.Pkgen(parameters[i,1], amp=parameters[i,0]), (self.input_shape[1], self.input_shape[2])) 
                      for i in range(parameters.shape[0])])
        
        
        if θ_fg is not None:
            if fg_repeat:
                θ_fg = np.repeat(
                  θ_fg[np.newaxis, :], 
                  parameters.shape[0], 
                  axis=0)
            fg = np.array([generate_field(self.distrib, self.Pkgen(θ_fg[i,1], 
                      amp=θ_fg[i,0]), (self.input_shape[1], self.input_shape[2])) 
                      for i in range(parameters.shape[0])])
        
            # return cosmo, fg separately
            if save_fg_copy:
                return d,fg     
            else:
                # add cosmo and fg
                d += fg; del fg
                return np.expand_dims(d, axis=1)
                
        # else return just cosmo
        else:
            d = np.expand_dims(d, axis=1)
            return d

    def generate_data(self, size="full"):
        self.check_selection(size)
        details = dict(
          input_shape=self.input_shape,
          n_params=self.n_params,
          n_summaries=self.n_summaries,
          n_s=self.n_s,
          n_d=self.n_d,
          θ_fid=self.θ_fid,
          θ_fg = self.θ_fg,
          δθ=self.δθ)
        
        # if foregrounds present, expand params
        if self.θ_fg is not None:
            fg_parameters = np.repeat(
                  self.θ_fg[np.newaxis, :], 
                  self.n_s, 
                  axis=0)
        else:
            fg_parameters = None

        # training base sims
        a_0 = self.simulator(
          parameters=np.repeat(
              self.θ_fid[np.newaxis, :], 
              self.n_s, 
              axis=0),
          θ_fg=fg_parameters,
          seed=self.training_seed,
          simulator_args={"input_shape": self.input_shape})
        # validation base sims
        a_1 = self.simulator(
          parameters=np.repeat(
              self.θ_fid[np.newaxis, :], 
              self.n_s, 
              axis=0),
          θ_fg=fg_parameters,
          seed=self.validation_seed,
          simulator_args={"input_shape": self.input_shape})

        # FOR NOW: TWO parameters: ONLY vary amp and power
        # training -amp
        b_0 = self.simulator(
          parameters=np.repeat(
              np.array([
                  self.θ_fid[0] - self.half_δθ[0],
                  self.θ_fid[1],
                  ])[np.newaxis, :], 
              self.n_d, 
              axis=0),
          θ_fg=fg_parameters,
          seed=self.training_seed,
          simulator_args={"input_shape": self.input_shape})
        # validation -amp
        b_1 = self.simulator(
          parameters=np.repeat(
              np.array([
                  self.θ_fid[0] - self.half_δθ[0],
                  self.θ_fid[1],
                  ])[np.newaxis, :], 
              self.n_d, 
              axis=0),
          θ_fg=fg_parameters,
          seed=self.validation_seed,
          simulator_args={"input_shape": self.input_shape})    
        # training +amp  
        c_0 = self.simulator(
          parameters=np.repeat(
              np.array([
                  self.θ_fid[0] + self.half_δθ[0],
                  self.θ_fid[1],
                  ])[np.newaxis, :], 
              self.n_d, 
              axis=0),
          θ_fg=fg_parameters,
          seed=self.training_seed,
          simulator_args={"input_shape": self.input_shape})
        # validation +amp
        c_1 = self.simulator(
          parameters=np.repeat(
              np.array([
                  self.θ_fid[0] + self.half_δθ[0],
                  self.θ_fid[1],
                  ])[np.newaxis, :], 
              self.n_d, 
              axis=0),
          θ_fg=fg_parameters,
          seed=self.validation_seed,
          simulator_args={"input_shape": self.input_shape})
        # training -power
        d_0 = self.simulator(
          parameters=np.repeat(
              np.array([
                  self.θ_fid[0], 
                  self.θ_fid[1] - self.half_δθ[1],
                  ])[np.newaxis, :], 
              self.n_d, 
              axis=0),
          θ_fg=fg_parameters,
          seed=self.training_seed,
          simulator_args={"input_shape": self.input_shape})
        # validation -power
        d_1 = self.simulator(
          parameters=np.repeat(
              np.array([
                  self.θ_fid[0], 
                  self.θ_fid[1] - self.half_δθ[1],
                  ])[np.newaxis, :], 
              self.n_d, 
              axis=0),
          θ_fg=fg_parameters,
          seed=self.validation_seed,
          simulator_args={"input_shape": self.input_shape})  
        # training +power     
        e_0 = self.simulator(
          parameters=np.repeat(
              np.array([
                  self.θ_fid[0], 
                  self.θ_fid[1] + self.half_δθ[1],
                  ])[np.newaxis, :], 
              self.n_d, 
              axis=0),
          θ_fg=fg_parameters,
          seed=self.training_seed,
          simulator_args={"input_shape": self.input_shape})
        # validation +power
        e_1 = self.simulator(
          parameters=np.repeat(
              np.array([
                  self.θ_fid[0], 
                  self.θ_fid[1] + self.half_δθ[1],
                  ])[np.newaxis, :], 
              self.n_d, 
              axis=0),
          θ_fg=fg_parameters,
          seed=self.validation_seed,
          simulator_args={"input_shape": self.input_shape}) 

        print('b0 shape : ', b_0.shape)
            
        f_0 = np.stack((np.stack((b_0, c_0)), 
                          np.stack((d_0, e_0)))
                        ).transpose(2, 1, 0, 3, 4, 5)
        f_1 = np.stack((np.stack((b_1, c_1)), 
                          np.stack((d_1, e_1)))
                        ).transpose(2, 1, 0, 3, 4, 5)

        result = (details, a_0, a_1, f_0, f_1)
          
        if size == "all":
            details["n_d_small"] = self.n_d_small
            result += (f_0[:self.n_d_small],
                    f_1[:self.n_d_small])
        elif size == "small":
            details["n_d"] = self.n_d_small
            result[-2] = f_0[:self.n_d_small]
            result[-1] = f_1[:self.n_d_small]

        return result
    
    def save(self, ftype="both", size="full", directory="data", record_size=0.01):
        self.check_ftype(ftype)
        result = self.generate_data(size=size)
        
        if (ftype=="both") or (ftype=="numpy"):
            np.savez("{}/details.npz".format(directory), result[0])
            np.save("{}/fiducial.npy".format(directory), result[1])
            np.save("{}/validation_fiducial.npy".format(directory), result[2])    
            np.save("{}/derivative.npy".format(directory), result[3])
            np.save("{}/validation_derivative.npy".format(directory), result[4])
            if size == "all":
                np.save("{}/derivative_small.npy".format(directory), result[5])
                np.save("{}/validation_derivative_small.npy".format(directory), result[6])

        if (ftype=="both") or (ftype=="tfrecords"):
            writer = TFRecords.TFRecords(record_size=record_size)
            
            writer.write_record(
                n_sims=result[0]["n_s"], 
                get_simulation=lambda x : self.get_fiducial(x, result[1]),
                fiducial=True, 
                directory="{}/tfrecords".format(directory))
            writer.write_record(
                n_sims=result[0]["n_s"], 
                get_simulation=lambda x : self.get_fiducial(x, result[2]),
                fiducial=True, 
                validation=True,
                directory="{}/tfrecords".format(directory))
            writer.write_record(
                n_sims=result[0]["n_d"], 
                get_simulation=lambda x, y, z : self.get_derivative(x, y, z, result[3]),
                fiducial=False,
                n_params=result[0]["n_params"],
                directory="{}/tfrecords".format(directory))
            writer.write_record(
                n_sims=result[0]["n_d"], 
                get_simulation=lambda x, y, z : self.get_derivative(x, y, z, result[4]),
                fiducial=False,
                n_params=result[0]["n_params"],
                validation=True,
                directory="{}/tfrecords".format(directory))
            if size == "all":
                writer.write_record(
                    n_sims=result[0]["n_d_small"], 
                    get_simulation=lambda x, y, z : self.get_derivative(x, y, z, result[5]),
                    fiducial=False,
                    n_params=result[0]["n_params"],
                    directory="{}/tfrecords".format(directory),
                    filename="derivative_small")
                writer.write_record(
                    n_sims=result[0]["n_d_small"], 
                    get_simulation=lambda x, y, z : self.get_derivative(x, y, z, result[6]),
                    fiducial=False,
                    n_params=result[0]["n_params"],
                    directory="{}/tfrecords".format(directory),
                    filename="derivative_small")
                
    def plot_data(self, data, pars=[0,1], plot_fg=False, ax=None, label=None,
                 cmap='jet'):
        if plot_fg:
            pars = np.squeeze(pars)
            labs = ['cosmo', 'foreground']
            
            cosmo,fg = data
            
            for i,d in enumerate([cosmo, fg]):
                
                fig = plt.figure()
                plt.imshow(np.squeeze(d), cmap=cmap)
                plt.colorbar(label=r'$\delta$')

                plt.title(r'%s field with $\theta_{\rm %s}=$(%.1f, %.1f)'%(labs[i], labs[i],
                                                                           pars[i][0], pars[i][1]))
                plt.show()
            
        else:
            pars = np.squeeze(pars)
            fig = plt.figure()

            plt.imshow(np.squeeze(data), cmap=cmap)
            plt.colorbar(label=r'$\delta$')


            plt.title(r'Gaussian field with $\theta_{\rm cosmo}=$(%.1f, %.1f)'%(pars[0], pars[1]))

        return fig,ax # for further modification


class GenerateCosmoFieldOneParam():
    def __init__(self, input_shape=(1,128,128), n_params=1, n_summaries=1, n_s=1000, n_d=1000, n_d_small=100,
                 θ_fid=np.array([1.0]), δθ=np.array([0.1]), θ_fg=None, training_seed=0,
                 validation_seed=1):
        
        self.input_shape = input_shape
        self.n_params = n_params
        self.n_summaries = n_summaries
        self.n_s = n_s
        self.n_d = n_d
        self.n_d_small = n_d_small
        self.θ_fid = θ_fid
        self.δθ = δθ
        self.half_δθ = δθ / 2.
        self.training_seed = training_seed
        self.validation_seed = validation_seed
        self.θ_fg = θ_fg

    def get_fiducial(self, seed, data):
        return data[seed]

    def get_derivative(self, seed, derivative, parameter, data):
        return data[seed, derivative, parameter]

    def check_selection(self, size):
        if size not in ["full", "all", "small"]:
            print("size must be `full`, `all` or `small` describing, respectively "
                  "whether just `n_d=n_s` is returned, or `n_d=n_s` and `n_d_small` "
                  "is returned, or `n_d=n_d_small` is returned.")
            sys.exit()
    
    def check_ftype(self, ftype):
        if ftype not in ["both", "numpy", "tfrecords"]:
            print("size must be `both`, `numpy` or `tfrecords` describing, respectively "
                  "whether both `numpy` and `tfrecords` files are saved, or just either one.")
            sys.exit()

    # Helper that generates power-law power spectrum
    def Pkgen(self, n, amp=1):
        def Pk(k):
            return amp*np.power(k, -n)

        return Pk

    # Draw samples from a normal distribution
    def distrib(self, shape):
        a = np.random.normal(loc=0, scale=1, size=shape)
        b = np.random.normal(loc=0, scale=1, size=shape)
        return a + 1j * b

    def simulator(self, parameters, θ_fg=None, seed=None, save_fg_copy=False,
                  fg_repeat=False,
                  simulator_args=None):
        # if self.input_shape[0] // 3 != 0:
        #     raise AssertionError ("input shape must be divisible by coordinate dimensions !")
        if seed is not None:
            np.random.seed(seed)
        if len(parameters.shape) == 1:
            parameters = parameters[np.newaxis, :]
        # if only looking at amplitude, fix power to 0.5
        if self.n_params == 1:
            parameters = np.repeat(parameters, 2, axis=1)
            parameters[:, 1] = np.ones_like(parameters[:, 1])*0.5

        
        d = np.array([generate_field(self.distrib, self.Pkgen(parameters[i,1], amp=parameters[i,0]), (self.input_shape[1], self.input_shape[2])) 
                      for i in range(parameters.shape[0])])
        
        
        if θ_fg is not None:
            if fg_repeat:
                θ_fg = np.repeat(
                  θ_fg[np.newaxis, :], 
                  parameters.shape[0], 
                  axis=0)
            fg = np.array([generate_field(self.distrib, self.Pkgen(θ_fg[i,1], 
                      amp=θ_fg[i,0]), (self.input_shape[1], self.input_shape[2])) 
                      for i in range(parameters.shape[0])])
        
            # return cosmo, fg separately
            if save_fg_copy:
                return d,fg     
            else:
                # add cosmo and fg
                d += fg; del fg
                return np.expand_dims(d, axis=1)
                
        # else return just cosmo
        else:
            d = np.expand_dims(d, axis=1)
            return d

    def generate_data(self, size="full"):
        self.check_selection(size)
        details = dict(
          input_shape=self.input_shape,
          n_params=self.n_params,
          n_summaries=self.n_summaries,
          n_s=self.n_s,
          n_d=self.n_d,
          θ_fid=self.θ_fid,
          θ_fg = self.θ_fg,
          δθ=self.δθ)
        
        # if foregrounds present, expand params
        if self.θ_fg is not None:
            fg_parameters = np.repeat(
                  self.θ_fg[np.newaxis, :], 
                  self.n_s, 
                  axis=0)
        else:
            fg_parameters = None

        # training base sims
        a_0 = self.simulator(
          parameters=np.repeat(
              self.θ_fid[np.newaxis, :], 
              self.n_s, 
              axis=0),
          θ_fg=fg_parameters,
          seed=self.training_seed,
          simulator_args={"input_shape": self.input_shape})
        # validation base sims
        a_1 = self.simulator(
          parameters=np.repeat(
              self.θ_fid[np.newaxis, :], 
              self.n_s, 
              axis=0),
          θ_fg=fg_parameters,
          seed=self.validation_seed,
          simulator_args={"input_shape": self.input_shape})

        # FOR NOW: ONE parameters=: ONLY vary amp
        # training -amp
        b_0 = self.simulator(
          parameters=np.repeat(
              np.array([
                  self.θ_fid[0] - self.half_δθ[0],
             #     self.θ_fid[1],
                  ])[np.newaxis, :], 
              self.n_d, 
              axis=0),
          θ_fg=fg_parameters,
          seed=self.training_seed,
          simulator_args={"input_shape": self.input_shape})
        # validation -amp
        b_1 = self.simulator(
          parameters=np.repeat(
              np.array([
                  self.θ_fid[0] - self.half_δθ[0],
              #    self.θ_fid[1],
                  ])[np.newaxis, :], 
              self.n_d, 
              axis=0),
          θ_fg=fg_parameters,
          seed=self.validation_seed,
          simulator_args={"input_shape": self.input_shape})    
        # training +amp  
        c_0 = self.simulator(
          parameters=np.repeat(
              np.array([
                  self.θ_fid[0] + self.half_δθ[0],
             #     self.θ_fid[1],
                  ])[np.newaxis, :], 
              self.n_d, 
              axis=0),
          θ_fg=fg_parameters,
          seed=self.training_seed,
          simulator_args={"input_shape": self.input_shape})
        # validation +amp
        c_1 = self.simulator(
          parameters=np.repeat(
              np.array([
                  self.θ_fid[0] + self.half_δθ[0],
             #     self.θ_fid[1],
                  ])[np.newaxis, :], 
              self.n_d, 
              axis=0),
          θ_fg=fg_parameters,
          seed=self.validation_seed,
          simulator_args={"input_shape": self.input_shape})

            
        f_0 = np.expand_dims(np.stack((b_0, c_0)).transpose(1, 0, 2, 3, 4), axis=-3)
        f_1 = np.expand_dims(np.stack((b_1, c_1)).transpose(1, 0, 2, 3, 4), axis=-3)

        result = (details, a_0, a_1, f_0, f_1)
          
        if size == "all":
            details["n_d_small"] = self.n_d_small
            result += (f_0[:self.n_d_small],
                    f_1[:self.n_d_small])
        elif size == "small":
            details["n_d"] = self.n_d_small
            result[-2] = f_0[:self.n_d_small]
            result[-1] = f_1[:self.n_d_small]

        return result
    
    def save(self, ftype="both", size="full", directory="data", record_size=0.01):
        self.check_ftype(ftype)
        result = self.generate_data(size=size)
        
        if (ftype=="both") or (ftype=="numpy"):
            np.savez("{}/details.npz".format(directory), result[0])
            np.save("{}/fiducial.npy".format(directory), result[1])
            np.save("{}/validation_fiducial.npy".format(directory), result[2])    
            np.save("{}/derivative.npy".format(directory), result[3])
            np.save("{}/validation_derivative.npy".format(directory), result[4])
            if size == "all":
                np.save("{}/derivative_small.npy".format(directory), result[5])
                np.save("{}/validation_derivative_small.npy".format(directory), result[6])

        if (ftype=="both") or (ftype=="tfrecords"):
            writer = TFRecords.TFRecords(record_size=record_size)
            
            writer.write_record(
                n_sims=result[0]["n_s"], 
                get_simulation=lambda x : self.get_fiducial(x, result[1]),
                fiducial=True, 
                directory="{}/tfrecords".format(directory))
            writer.write_record(
                n_sims=result[0]["n_s"], 
                get_simulation=lambda x : self.get_fiducial(x, result[2]),
                fiducial=True, 
                validation=True,
                directory="{}/tfrecords".format(directory))
            writer.write_record(
                n_sims=result[0]["n_d"], 
                get_simulation=lambda x, y, z : self.get_derivative(x, y, z, result[3]),
                fiducial=False,
                n_params=result[0]["n_params"],
                directory="{}/tfrecords".format(directory))
            writer.write_record(
                n_sims=result[0]["n_d"], 
                get_simulation=lambda x, y, z : self.get_derivative(x, y, z, result[4]),
                fiducial=False,
                n_params=result[0]["n_params"],
                validation=True,
                directory="{}/tfrecords".format(directory))
            if size == "all":
                writer.write_record(
                    n_sims=result[0]["n_d_small"], 
                    get_simulation=lambda x, y, z : self.get_derivative(x, y, z, result[5]),
                    fiducial=False,
                    n_params=result[0]["n_params"],
                    directory="{}/tfrecords".format(directory),
                    filename="derivative_small")
                writer.write_record(
                    n_sims=result[0]["n_d_small"], 
                    get_simulation=lambda x, y, z : self.get_derivative(x, y, z, result[6]),
                    fiducial=False,
                    n_params=result[0]["n_params"],
                    directory="{}/tfrecords".format(directory),
                    filename="derivative_small")
                
    def plot_data(self, data, pars=[0,1], plot_fg=False, ax=None, label=None,
                 cmap='jet'):
        if plot_fg:
            pars = np.squeeze(pars)

            labs = ['cosmo', 'foreground']
            
            cosmo,fg = data
            
            for i,d in enumerate([cosmo, fg]):
                
                fig = plt.figure()
                plt.imshow(np.squeeze(d), cmap=cmap)
                plt.colorbar(label=r'$\delta$')

                plt.title(r'%s field with $\theta_{\rm %s}=$(%.1f, %.1f)'%(labs[i], labs[i],
                                                                           pars[i][0], pars[i][1]))
                plt.show()
            
        else:
            pars = np.squeeze(pars)
            fig = plt.figure()

            plt.imshow(np.squeeze(data), cmap=cmap)
            plt.colorbar(label=r'$\delta$')


            plt.title(r'Gaussian field with $\theta_{\rm cosmo}=$(%.1f, %.1f)'%(pars[0], pars[1]))

        return fig,ax # for further modification
