from jax import linear_util as lu
from jax.tree_util import tree_map, tree_transpose, tree_structure
from jax.util import partial
from jax.api import argnums_partial, _vjp, _jvp, vmap,_std_basis, \
    _check_callable, _unravel_array_into_pytree, \
    _check_input_dtype_jacrev, _check_output_dtype_jacrev, \
    _check_input_dtype_jacfwd, _check_output_dtype_jacfwd
from typing import Callable, Union, Sequence

def value_and_jacrev(fun: Callable, argnums: Union[int, Sequence[int]] = 0,
           holomorphic: bool = False, allow_int: bool = False) -> Callable:
  """Jacobian of ``fun`` evaluated row-by-row using reverse-mode AD.

  Args:
    fun: Function whose Jacobian is to be computed.
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default ``0``).
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. Default False.
    allow_int: Optional, bool. Whether to allow differentiating with
      respect to integer valued inputs. The gradient of an integer input will
      have a trivial vector-space dtype (float0). Default False.

  Returns:
    A function with the same arguments as ``fun``, that evaluates the Jacobian of
    ``fun`` using reverse-mode automatic differentiation.

  >>> import jax
  >>> import jax.numpy as jnp
  >>>
  >>> def f(x):
  ...   return jnp.asarray(
  ...     [x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jnp.sin(x[0])])
  ...
  >>> print(jax.jacrev(f)(jnp.array([1., 2., 3.])))
  [[ 1.       0.       0.     ]
   [ 0.       0.       5.     ]
   [ 0.      16.      -2.     ]
   [ 1.6209   0.       0.84147]]
  """
  _check_callable(fun)

  def jacfun(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = argnums_partial(f, argnums, args)
    tree_map(partial(_check_input_dtype_jacrev, holomorphic, allow_int), dyn_args)
    y, pullback = _vjp(f_partial, *dyn_args)
    tree_map(partial(_check_output_dtype_jacrev, holomorphic), y)
    jac = vmap(pullback)(_std_basis(y))
    jac = jac[0] if isinstance(argnums, int) else jac
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    jac = tree_map(partial(_unravel_array_into_pytree, y, 0), jac)
    return y, tree_transpose(tree_structure(example_args), tree_structure(y), jac)

  return jacfun

def value_and_jacfwd(fun: Callable, argnums: Union[int, Sequence[int]] = 0,
           holomorphic: bool = False) -> Callable:
  """Jacobian of ``fun`` evaluated column-by-column using forward-mode AD.
  Args:
    fun: Function whose Jacobian is to be computed.
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default ``0``).
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. Default False.
  Returns:
    A function with the same arguments as ``fun``, that evaluates the Jacobian of
    ``fun`` using forward-mode automatic differentiation.
  >>> import jax
  >>> import jax.numpy as jnp
  >>>
  >>> def f(x):
  ...   return jnp.asarray(
  ...     [x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jnp.sin(x[0])])
  ...
  >>> print(jax.jacfwd(f)(jnp.array([1., 2., 3.])))
  [[ 1.       0.       0.     ]
   [ 0.       0.       5.     ]
   [ 0.      16.      -2.     ]
   [ 1.6209   0.       0.84147]]
  """
  _check_callable(fun)

  def jacfun(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = argnums_partial(f, argnums, args)
    tree_map(partial(_check_input_dtype_jacfwd, holomorphic), dyn_args)
    pushfwd = partial(_jvp, f_partial, dyn_args)
    y, jac = vmap(pushfwd, out_axes=(None, -1))(_std_basis(dyn_args))
    tree_map(partial(_check_output_dtype_jacfwd, holomorphic), y)
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    return y, tree_map(partial(_unravel_array_into_pytree, example_args, -1), jac)

  return jacfun
