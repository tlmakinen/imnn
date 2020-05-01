import argparse
import sys
import numpy as np
from IMNN.utils import TFRecords

class GenerateGaussianNoise():
    def __init__(self, input_shape=(10,), n_params=2, n_summaries=2, n_s=1000, n_d=1000, n_d_small=100,
                 fiducial=np.array([0., 1.]), delta=np.array([0.1, 0.1]), training_seed=0,
                 validation_seed=1):
        self.input_shape = input_shape
        self.n_params = n_params
        self.n_summaries = n_summaries
        self.n_s = n_s
        self.n_d = n_d
        self.n_d_small = n_d_small
        self.fiducial = fiducial
        self.delta = delta
        self.training_seed = training_seed
        self.validation_seed = validation_seed

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
        
    def generate_data(self, size="full"):
        self.check_selection(size)
        details = dict(
            input_shape=self.input_shape,
            n_params=self.n_params,
            n_summaries=self.n_summaries,
            n_s=self.n_s,
            n_d=self.n_d,
            fiducial=self.fiducial,
            delta=(2. * self.delta))
        
        np.random.seed(self.training_seed)
        a_0 = np.random.normal(
            self.fiducial[0], 
            np.sqrt(self.fiducial[1]), 
            (self.n_s,) + self.input_shape)
        np.random.seed(self.validation_seed)
        a_1 = np.random.normal(
            self.fiducial[0], 
            np.sqrt(self.fiducial[1]), 
            (self.n_s,) + self.input_shape)

        np.random.seed(self.training_seed)
        b_0 = np.random.normal(
            self.fiducial[0] - self.delta[0], 
            np.sqrt(self.fiducial[1]), 
            (self.n_d,) + self.input_shape)
        np.random.seed(self.validation_seed)
        b_1 = np.random.normal(
            self.fiducial[0] - self.delta[0], 
            np.sqrt(self.fiducial[1]), 
            (self.n_d,) + self.input_shape)
        np.random.seed(self.training_seed)
        c_0 = np.random.normal(
            self.fiducial[0] + self.delta[0], 
            np.sqrt(self.fiducial[1]), 
            (self.n_d,) + self.input_shape)
        np.random.seed(self.validation_seed)
        c_1 = np.random.normal(
            self.fiducial[0] + self.delta[0], 
            np.sqrt(self.fiducial[1]), 
            (self.n_d,) + self.input_shape)
        np.random.seed(self.training_seed)
        d_0 = np.random.normal(
            self.fiducial[0], 
            np.sqrt(self.fiducial[1] - self.delta[1]), 
            (self.n_d,) + self.input_shape)
        np.random.seed(self.validation_seed)
        d_1 = np.random.normal(
            self.fiducial[0], 
            np.sqrt(self.fiducial[1] - self.delta[1]), 
            (self.n_d,) + self.input_shape)
        np.random.seed(self.training_seed)
        e_0 = np.random.normal(
            self.fiducial[0], 
            np.sqrt(self.fiducial[1] + self.delta[1]), 
            (self.n_d,) + self.input_shape)
        np.random.seed(self.validation_seed)
        e_1 = np.random.normal(
            self.fiducial[0], 
            np.sqrt(self.fiducial[1] + self.delta[1]), 
            (self.n_d,) + self.input_shape)

        f_0 = np.stack((np.stack((b_0, c_0)), 
                        np.stack((d_0, e_0)))
                      ).transpose(2, 1, 0, 3)
        f_1 = np.stack((np.stack((b_1, c_1)), 
                        np.stack((d_1, e_1)))
                      ).transpose(2, 1, 0, 3)

        result = (details, a_0, a_1, f_0, f_1)
            
        if size == "all":
            details["n_d_small"] = self.n_d_small
            result += (f_0[:self.n_d_small],
                        f_1[:self.n_d_small])
        if size == "small":
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
    
def main(args):
    data = GenerateGaussianNoise(
        input_shape=tuple(args.input_shape),
        n_params=args.n_params, 
        n_summaries=args.n_summaries, 
        n_s=args.n_s,
        n_d=args.n_d,
        n_d_small=args.n_d_small,
        fiducial=np.array(args.fiducial), 
        delta=np.array(args.delta), 
        training_seed=args.training_seed,
        validation_seed=args.validation_seed)
    data.save(
        ftype=args.ftype,
        size=args.size,
        directory=args.directory,
        record_size=args.record_size)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_shape', nargs='*', default=[10], type=int)
    parser.add_argument('--n_params', nargs='?', const=1, default=2, type=int)
    parser.add_argument('--n_summaries', nargs='?', const=1, default=2, type=int)
    parser.add_argument('--n_s', nargs='?', const=1, default=1000, type=int)
    parser.add_argument('--n_d', nargs='?', const=1, default=1000, type=int)
    parser.add_argument('--n_d_small', nargs='?', const=1, default=100, type=int)
    parser.add_argument('--fiducial', nargs='*', default=[0, 1], type=float)
    parser.add_argument('--delta', nargs='*', default=[0.1, 0.1], type=float)
    parser.add_argument('--training_seed', nargs='?', const=1, default=0, type=int)
    parser.add_argument('--validation_seed', nargs='?', const=1, default=1, type=int)
    parser.add_argument('--ftype', nargs='?', const=1, default="both", type=str)
    parser.add_argument('--size', nargs='?', const=1, default="all", type=str)
    parser.add_argument('--directory', nargs='?', const=1, default="data", type=str)
    parser.add_argument('--record_size', nargs='?', const=1, default=0.01, type=float)
    args = parser.parse_args()
    main(args)