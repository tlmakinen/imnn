import inspect
import jax.numpy as np

class ShapeError(Exception):
    pass

class FunctionError(Exception):
    pass

def check_type(input, target_type, name, shape=None):
    if input is None:
        raise ValueError("`{}` is None".format(name))
    elif type(input) is not target_type:
        raise TypeError("`{}` must be type {} but is {}".format(
            name, target_type, type(input)))
    elif shape is not None:
        try:
            input_shape = input.shape
            if input_shape != shape:
                raise ShapeError("`{}` must have shape {} but has shape {}" \
                    .format(name, shape, input_shape))
            else:
                return input
        except:
            input_shape = len(input)
            if input_shape != shape:
                raise ShapeError("`{}` must have lenth {} but has length {}" \
                    .format(name, shape, input_shape))
            else:
                return input
    else:
        return input

def check_simulator(simulator):
    if simulator is None:
        raise ValueError("no `simulator`")
    elif not callable(simulator):
        raise TypeError("`simulator` not callable")
    else:
        if len(inspect.signature(simulator).parameters) != 2:
            raise FunctionError("`simulator` must take two arguments, a " +
                                 "JAX prng and simulator parameters.")
        return simulator

def check_model(model):
    if model is None:
        raise ValueError("no `model`")
    elif not tuple(model):
        raise TypeError("`model` not a tuple of callables")
    else:
        if len(model) != 2:
            raise ShapeError("`model` must be a tuple of two functions. The " +
                            "first for initialising the model and the " +
                            "second to call the model")
        if len(inspect.signature(model[0]).parameters) != 2:
            raise FunctionError("first element of `model` must take two " +
                            "arguments, {}".format(
                inspect.signature(model[0])))
        if len(inspect.signature(model[1]).parameters) != 3:
            raise FunctionError("second element of `model` must take three " +
                            "arguments, {}".format(
                inspect.signature(model[1])))
        return model

def check_optimiser(optimiser):
    if optimiser is None:
        raise ValueError("no `optimiser`")
    else:
        try:
            length = len(optimiser)
        except:
            raise TypeError("`optimiser` does not have a length")
        if length != 3:
            raise ShapeError("`optimiser` must be a tuple of three functions."+
                            " The first for initialising the state, the " +
                            "second to update the state and the third " +
                            "to get parameters from the state.")
        if len(inspect.signature(optimiser[0]).parameters) != 1:
            raise FunctionError("first element of `optimiser` must take " +
                                 "one argument, {}".format(
                inspect.signature(optimiser[0])))
        if len(inspect.signature(optimiser[1]).parameters) != 3:
            raise FunctionError("second element of `optimiser` must take " +
                                 "three arguments, {}".format(
                inspect.signature(optimiser[1])))
        if len(inspect.signature(optimiser[2]).parameters) != 1:
            raise FunctionError("last element of `optimiser` must take one " +
                            "argument, {}".format(
                inspect.signature(optimiser[2])))
        return optimiser

def check_splitting(size, name, n_devices, n_per_device):
    if (size / (n_devices * n_per_device)
            != float(size // (n_devices * n_per_device))):
        raise ValueError("`{}` of {} will not split evenly between ".format(
                         name, size) + "{} devices when calculating ".format(
                         n_devices) + "{} per device.".format(
                         n_per_device))

def check_input(input, shape, name):
    if input is None:
        raise ValueError("no `{}` simulations".format(name))
    elif type(input) != type(np.empty(shape)):
        raise TypeError("`{}` does has type {} and not {}".format(
                        name, type(input), type(np.empty(shape))))
    else:
        if input.shape != shape:
            raise ShapeError("`{}` should have shape {} but has {}".format(
                             name, shape, input.shape))
    return input
