from IMNN.experimental.jax.utils.utils import *
from IMNN.experimental.jax.utils import value_and_jacrev
from IMNN.experimental.jax.imnn import SimulatorIMNN, AggregatedSimulatorIMNN,\
    GradientIMNN, NumericalGradientIMNN
from datetime import datetime
import jax
import jax.numpy as np
import jax.experimental.stax as stax
import jax.experimental.optimizers as optimizers
from functools import partial
import itertools
import inspect
import tqdm
rng = jax.random.PRNGKey(0)

def bad_simulator(rng, θ, extra_argument):
    print("This simulator has too many arguments")

def simulator(key, θ):
    μ, Σ = θ
    return μ + np.sqrt(Σ) * jax.random.normal(key, shape=(10,))

def simulator_gradient(key, θ):
    return value_and_jacrev(simulator, argnums=1)(key, θ)

rng, *keys = jax.random.split(rng, num=1000+1)
fiducial, derivatives = jax.vmap(
    partial(simulator_gradient, θ=np.array([0., 1.])))(np.array(keys))
numerical_derivatives = np.stack([
    np.stack([
        jax.vmap(partial(simulator, θ=(-0.05, 1.)))(np.array(keys)),
        jax.vmap(partial(simulator, θ=(0., -0.95)))(np.array(keys))],
        1),
    np.stack([
        jax.vmap(partial(simulator, θ=(0.05, 1.)))(np.array(keys)),
        jax.vmap(partial(simulator, θ=(0., 1.05)))(np.array(keys))],
        1)],
    1)
rng, *keys = jax.random.split(rng, num=1000+1)
validation_fiducial, validation_derivatives = jax.vmap(
    partial(simulator_gradient, θ=np.array([0., 1.])))(np.array(keys))
validation_numerical_derivatives = np.stack([
    np.stack([
        jax.vmap(partial(simulator, θ=(-0.05, 1.)))(np.array(keys)),
        jax.vmap(partial(simulator, θ=(0., -0.95)))(np.array(keys))],
        1),
    np.stack([
        jax.vmap(partial(simulator, θ=(0.05, 1.)))(np.array(keys)),
        jax.vmap(partial(simulator, θ=(0., 1.05)))(np.array(keys))],
        1)],
    1)

def test(
    n_s=[1000, 1000., None],
    n_d=[1000, 100, 1000., None],
    n_params=[2, 5, 2., None],
    n_summaries=[2, 5, 2., None],
    input_shape=[(10,), np.array([10,]), None],
    θ_fid=[np.array([0., 1.]), [0., 1.], None],
    initial_model_key=[jax.random.PRNGKey(1), "wrong type", None],
    model=[
        stax.serial(
            stax.Dense(128),
            stax.LeakyRelu,
            stax.Dense(128),
            stax.LeakyRelu,
            stax.Dense(128),
            stax.LeakyRelu,
            stax.Dense(2)),
        "wrong type", None],
    optimiser=[optimizers.adam(step_size=1e-3), "wrong type", None],
    fitting_key=[jax.random.PRNGKey(2), "wrong type", None],
    λ=[10., "wrong type", None],
    ϵ=[0.1, "wrong type", None],
    simulator_fn=[simulator, bad_simulator, "wrong type", None],
    devices=[jax.devices(), "wrong type", None],
    n_per_device=[1000, 500, 11, "wrong type", None],
    fiducial_data=[fiducial, fiducial.T, "wrong type", None],
    derivative_data=[
        derivatives,
        np.swapaxes(derivatives, -2, -1),
        "wrong type", None],
    derivative_cut=[1000, 100],
    validation_fiducial_data=[
        validation_fiducial,
        validation_fiducial.T,
        "wrong type", None],
    validation_derivative_data=[
        validation_derivatives,
        np.swapaxes(validation_derivatives, -2, -1),
        "wrong type", None],
    numerical_derivative_data=[
        numerical_derivatives,
        np.swapaxes(numerical_derivatives, -2, -1),
        "wrong type", None],
    validation_numerical_derivative_data=[
        validation_numerical_derivatives,
        np.swapaxes(validation_numerical_derivatives, -2, -1),
        "wrong type", None],
    δθ=[np.array([0.1, 0.1]), [0.1, 0.1], None]):

    print("Testing SimulatorIMNN")
    testSimulatorIMNN(
        n_s=n_s,
        n_d=n_d,
        n_params=n_params,
        n_summaries=n_summaries,
        input_shape=input_shape,
        θ_fid=θ_fid,
        initial_model_key=initial_model_key,
        model=model,
        optimiser=optimiser,
        fitting_key=fitting_key,
        λ=λ,
        ϵ=ϵ,
        simulator_fn=simulator_fn)
    '''
    print("Testing AggregatedSimulatorIMNN")
    testAggregatedSimulatorIMNN(
        n_s=n_s,
        n_d=n_d,
        n_params=n_params,
        n_summaries=n_summaries,
        input_shape=input_shape,
        θ_fid=θ_fid,
        initial_model_key=initial_model_key,
        model=model,
        optimiser=optimiser,
        fitting_key=fitting_key,
        λ=λ,
        ϵ=ϵ,
        simulator_fn=simulator_fn,
        devices=devices,
        n_per_device=n_per_device)
    print("Testing GradientIMNN")
    testGradientIMNN(
        n_s=n_s,
        n_d=n_d,
        n_params=n_params,
        n_summaries=n_summaries,
        input_shape=input_shape,
        θ_fid=θ_fid,
        initial_model_key=initial_model_key,
        model=model,
        optimiser=optimiser,
        fitting_key=fitting_key,
        λ=λ,
        ϵ=ϵ,
        fiducial_data=fiducial_data,
        derivative_data=derivative_data,
        derivative_cut=derivative_cut,
        validation_fiducial_data=validation_fiducial_data,
        validation_derivative_data=validation_derivative_data)
    print("Testing NumericalGradientIMNN")
    testNumericalGradientIMNN(
        n_s=n_s,
        n_d=n_d,
        n_params=n_params,
        derivative_cut=derivative_cut,
        n_summaries=n_summaries,
        input_shape=input_shape,
        θ_fid=θ_fid,
        initial_model_key=initial_model_key,
        model=model,
        optimiser=optimiser,
        fitting_key=fitting_key,
        λ=λ,
        ϵ=ϵ,
        fiducial_data=fiducial_data,
        derivative_data=derivative_data,
        numerical_derivative_cut=numerical_derivative_cut,
        validation_fiducial_data=validation_fiducial_data,
        validation_numerical_derivative_data= \
            validation_numerical_derivative_data)
    '''

def testSimulatorIMNN(
        n_s=[1000, 1000., None],
        n_d=[1000, 100, 1000., None],
        n_params=[2, 5, 2., None],
        n_summaries=[2, 5, 2., None],
        input_shape=[(10,), np.array([10,]), None],
        θ_fid=[np.array([0., 1.]), [0., 1.], None],
        initial_model_key=[jax.random.PRNGKey(1), "wrong type", None],
        model=[
            stax.serial(
                stax.Dense(128),
                stax.LeakyRelu,
                stax.Dense(128),
                stax.LeakyRelu,
                stax.Dense(128),
                stax.LeakyRelu,
                stax.Dense(2)),
            "wrong type", None],
        optimiser=[optimizers.adam(step_size=1e-3), "wrong type", None],
        fitting_key=[jax.random.PRNGKey(2), "wrong type", None],
        λ=[10., "wrong type", None],
        ϵ=[0.1, "wrong type", None],
        simulator_fn=[simulator, bad_simulator, "wrong type", None],):
    bar = tqdm.tqdm(
            itertools.product(
                n_s, n_d, n_params, n_summaries, input_shape, θ_fid,
                initial_model_key, model, optimiser, fitting_key, λ, ϵ,
                simulator_fn),
            leave=False)
    for s, d, p, sum, shape, fid, mod_key, mod, opt, fit_key, l, e, sim in bar:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if mod is not None:
            if type(mod) == tuple:
                mod_print = len(mod)
            else:
                mod_print = type(mod)
        else:
            mod_print = None
        if opt is not None:
            try:
                length = len(opt)
                opt_print = length
            except:
                opt_print = type(opt)
        else:
            opt_print = None
        if sim is not None:
            if callable(sim):
                sim_print = inspect.signature(sim)
            else:
                sim_print = type(sim)
        else:
            sim_print = None
        bar.set_postfix(
            n_s=s,
            n_d=d,
            n_params=p,
            n_summaries=sum,
            input_shape=shape,
            θ_fid=fid,
            initial_model_key=mod_key,
            model=mod_print,
            optimiser=opt_print,
            fitting_key=fit_key,
            λ=l,
            ϵ=e,
            simulator=sim_print)

        try:
            imnn = SimulatorIMNN(
                n_s=s, n_d=d, n_summaries=sum, n_params=n_params,
                input_shape=shape, θ_fid=fid, key=mod_key, model=mod,
                optimiser=opt, simulator=sim)
            imnn.fit(λ=l, ϵ=e, rng=fit_key)
            imnn.training_plot(
                expected_detF=50,
                filename="figures/SimulatorIMNN_{}.png".format(timestamp))
        except ValueError as e:
            pass
        except TypeError as e:
            pass
        except ShapeError as e:
            pass
        except FunctionError as e:
            pass
        except Exception as e:
            print(e)
            break

def testAggregatedSimulatorIMNN(
        n_s=[1000, 1000., "wrong type", None],
        n_d=[1000, 100, 1000., "wrong type", None],
        n_params=[2, 5, 2., "wrong type", None],
        n_summaries=[2, 5, 2., "wrong type", None],
        input_shape=[(10,), np.array([10,]), "wrong type", None],
        θ_fid=[np.array([0., 1.]), [0., 1.], "wrong type", None],
        initial_model_key=[jax.random.PRNGKey(1), "wrong type", None],
        model=[
            stax.serial(
                stax.Dense(128),
                stax.LeakyRelu,
                stax.Dense(128),
                stax.LeakyRelu,
                stax.Dense(128),
                stax.LeakyRelu,
                stax.Dense(2)),
            "wrong type", None],
        optimiser=[optimizers.adam(step_size=1e-3), "wrong type", None],
        fitting_key=[jax.random.PRNGKey(2), "wrong type", None],
        λ=[10., "wrong type", None],
        ϵ=[0.1, "wrong type", None],
        simulator_fn=[simulator, bad_simulator, "wrong type", None],
        devices=[jax.devices(), 1, "wrong type", None],
        n_per_device=[1000, 500, 11, "wrong type", None]):
    for s, d, sum, shape, fid, mod_key, mod, opt, fit_key, l, e, sim, dev, n \
            in itertools.product(
                n_s, n_d, n_summaries, input_shape, θ_fid, initial_model_key,
                model, optimiser, fitting_key, λ, ϵ, simulator_fn, devices,
                n_per_device):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        print(timestamp)
        print("n_s={}".format(s))
        print("n_d={}".format(d))
        print("n_summaries={}".format(sum))
        print("input_shape={}".format(shape))
        print("θ_fid={}".format(fid))
        print("initial_model_key={}".format(mod_key))
        print("model={}".format(mod))
        print("optimiser={}".format(opt))
        print("fitting_key={}".format(fit_key))
        print("λ={}".format(l))
        print("ϵ={}".format(e))
        if sim is not None:
            print("simulator_fn={}".format(inspect.signature(sim)))
        else:
            print("simulator_fn={}".format(sim))
        print("devices={}".format(dev))
        print("n_per_device={}".format(n))
        try:
            imnn = AggregatedSimulatorIMNN(
                n_s=s, n_d=d, n_summaries=sum, input_shape=shape, θ_fid=fid,
                key=mod_key, model=mod, optimiser=opt, simulator=sim,
                devices=devices)
            imnn.fit(λ=l, ϵ=e, rng=fit_key)
            imnn.training_plot(
                expected_detF=50,
                filename="figures/AggregatedSimulatorIMNN_{}.png".format(
                    timestamp))
        except:
            print("Exception worked")

def testGradientIMNN(
        n_s=[1000, 1000., "wrong type", None],
        n_d=[1000, 100, 1000., "wrong type", None],
        n_params=[2, 5, 2., "wrong type", None],
        n_summaries=[2, 5, 2., "wrong type", None],
        input_shape=[(10,), np.array([10,]), "wrong type", None],
        θ_fid=[np.array([0., 1.]), [0., 1.], None],
        initial_model_key=[jax.random.PRNGKey(1), "wrong type", None],
        model=[
            stax.serial(
                stax.Dense(128),
                stax.LeakyRelu,
                stax.Dense(128),
                stax.LeakyRelu,
                stax.Dense(128),
                stax.LeakyRelu,
                stax.Dense(2)),
            "wrong type", None],
        optimiser=[optimizers.adam(step_size=1e-3), "wrong type", None],
        fitting_key=[jax.random.PRNGKey(2), "wrong type", None],
        λ=[10., "wrong type", None],
        ϵ=[0.1, "wrong type", None],
        fiducial_data=[fiducial, fiducial.T, "wrong type", None],
        derivative_data=[
            derivatives,
            np.swapaxes(derivatives, -2, -1),
            "wrong type", None],
        derivative_cut=[1000, 100],
        validation_fiducial_data=[
            validation_fiducial,
            validation_fiducial.T,
            "wrong type", None],
        validation_derivative_data=[
            validation_derivatives,
            np.swapaxes(validation_derivatives, -2, -1),
            "wrong type", None]):
    for s, d, sum, shape, fid, mod_key, mod, opt, fit_key, l, e, f, dd, c, vf,\
        vdd in itertools.product(
            n_s, n_d, n_summaries, input_shape, θ_fid, initial_model_key,
            model, optimiser, fitting_key, λ, ϵ, fiducial_data,
            derivative_data, derivative_cut, validation_fiducial_data,
            validation_derivative_data):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        print(timestamp)
        print("n_s={}".format(s))
        print("n_d={}".format(d))
        print("n_summaries={}".format(sum))
        print("input_shape={}".format(shape))
        print("θ_fid={}".format(fid))
        print("initial_model_key={}".format(mod_key))
        print("model={}".format(mod))
        print("optimiser={}".format(opt))
        print("fitting_key={}".format(fit_key))
        print("λ={}".format(l))
        print("ϵ={}".format(e))
        print("derivative_cut={}".format(c))
        if f is not None:
            print("fiducial={}".format(f.shape))
        else:
            print("fiducial={}".format(f))
        if dd is not None:
            dd = dd[:c]
            print("derivative={}".format(dd.shape))
        else:
            print("derivative={}".format(dd))
        if vf is not None:
            print("validation_fiducial={}".format(vf.shape))
        else:
            print("validation_fiducial={}".format(vf))
        if vdd is not None:
            vdd = vdd[:c]
            print("validation_derivative={}".format(vdd.shape))
        else:
            print("validation_derivative={}".format(vdd))
        try:
            imnn = GradientIMNN(
                n_s=s, n_d=d, n_summaries=sum, input_shape=shape, θ_fid=fid,
                key=mod_key, model=mod, optimiser=opt, fiducial=f,
                derivative=dd, validation_fiducial=vf,
                validation_derivative=vdd)
            imnn.fit(λ=l, ϵ=e, rng=fit_key)
            imnn.training_plot(
                expected_detF=50,
                filename="figures/GradientIMNN_{}.png".format(timestamp))
        except:
            print("Exception worked")

def testNumericalGradientIMNN(
        n_s=[1000, 1000., "wrong type", None],
        n_d=[1000, 100, 1000., "wrong type", None],
        n_params=[2, 5, 2., "wrong type", None],
        n_summaries=[2, 5, 2., "wrong type", None],
        input_shape=[(10,), np.array([10,]), "wrong type", None],
        θ_fid=[np.array([0., 1.]), [0., 1.], "wrong type", None],
        initial_model_key=[jax.random.PRNGKey(1), "wrong type", None],
        model=[
            stax.serial(
                stax.Dense(128),
                stax.LeakyRelu,
                stax.Dense(128),
                stax.LeakyRelu,
                stax.Dense(128),
                stax.LeakyRelu,
                stax.Dense(2)),
            "wrong type", None],
        optimiser=[optimizers.adam(step_size=1e-3), "wrong type", None],
        fitting_key=[jax.random.PRNGKey(2), "wrong type", None],
        λ=[10., "wrong type", None],
        ϵ=[0.1, "wrong type", None],
        fiducial_data=[fiducial, fiducial.T, "wrong type", None],
        derivative_cut=[1000, 100],
        validation_fiducial_data=[
            validation_fiducial,
            validation_fiducial.T,
            "wrong type", None],
        numerical_derivative_data=[
            numerical_derivatives,
            np.swapaxes(numerical_derivatives, -2, -1),
            "wrong type", None],
        validation_numerical_derivative_data=[
            validation_numerical_derivatives,
            np.swapaxes(validation_numerical_derivatives, -2, -1),
            "wrong type", None],
        δθ=[np.array([0.1, 0.1]), [0.1, 0.1], "wrong type", None]):
    for s, d, sum, shape, fid, mod_key, mod, opt, fit_key, l, e, f, dd, c, \
        vf, vdd, dt in itertools.product(
            n_s, n_d, n_summaries, input_shape, θ_fid, initial_model_key,
            model, optimiser, fitting_key, λ, ϵ, fiducial_data,
            numerical_derivative_data, derivative_cut,
            validation_fiducial_data, validation_numerical_derivative_data,
            δθ):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        print(timestamp)
        print("n_s={}".format(s))
        print("n_d={}".format(d))
        print("n_summaries={}".format(sum))
        print("input_shape={}".format(shape))
        print("θ_fid={}".format(fid))
        print("initial_model_key={}".format(mod_key))
        print("model={}".format(mod))
        print("optimiser={}".format(opt))
        print("fitting_key={}".format(fit_key))
        print("λ={}".format(l))
        print("ϵ={}".format(e))
        print("derivative_cut={}".format(c))
        print("δθ={}".format(dt))
        if f is not None:
            print("fiducial={}".format(f.shape))
        else:
            print("fiducial={}".format(f))
        if dd is not None:
            dd = dd[:c]
            print("derivative={}".format(dd.shape))
        else:
            print("derivative={}".format(dd))
        if vf is not None:
            print("validation_fiducial={}".format(vf.shape))
        else:
            print("validation_fiducial={}".format(vf))
        if vdd is not None:
            vdd = vdd[:c]
            print("validation_derivative={}".format(vdd.shape))
        else:
            print("validation_derivative={}".format(vdd))
        try:
            imnn = GradientIMNN(
                n_s=s, n_d=d, n_summaries=sum, input_shape=shape, θ_fid=fid,
                key=mod_key, model=mod, optimiser=opt, fiducial=f,
                derivative=dd, validation_fiducial=vf,
                validation_derivative=vdd)
            imnn.fit(λ=l, ϵ=e, rng=fit_key)
            imnn.training_plot(
                expected_detF=50,
                filename="figures/NumericalGradientIMNN_{}.png".format(
                    timestamp))
        except:
            print("Exception worked")

if __name__ == "__main__":
    test()
