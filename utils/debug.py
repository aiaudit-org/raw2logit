import torch
import numpy as np
import inspect
from functools import reduce, wraps
from collections.abc import Iterable
from IPython import embed

try:
    get_ipython()  # pylint: disable=undefined-variable
    interactive_notebook = True
except:
    interactive_notebook = False

_NONE = "__UNSET_VARIABLE__"


def debug_init():
    debug.disable = False
    debug.silent = False
    debug.verbose = 2
    debug.expand_ignore = ["DataLoader", "Dataset", "Subset"]
    debug.max_expand = 10
    debug.show_tensor = False
    debug.raise_exception = True
    debug.full_stack = True
    debug.restore_defaults_on_exception = not interactive_notebook
    debug._indent = 0
    debug._stack = ""

    debug.embed = embed
    debug.show = debug_show
    debug.pause = debug_pause


def debug_pause():
    input("Press Enter to continue...")


def debug(*args, assert_true=False):
    """Decorator for debugging functions and tensors.
    Will throw an exception as soon as a nan is encountered.
    If used on iterables, these will be expanded and also searched for nans.
    Usage:
        debug(x)
    Or:
        @debug
        def function():
            ...
    If used as a function wrapper, all arguments will be searched and printed.
    """

    single_arg = len(args) == 1

    if debug.disable:
        return args[0] if single_arg else None

    try:
        call_line = ''.join(inspect.stack()[1][4]).strip()
    except:
        call_line = '...'
    used_as_wrapper = 'def ' == call_line[:4]
    expect_return_arg = single_arg and 'debug' in call_line and call_line.split('debug')[0].strip() != ''
    is_func = single_arg and hasattr(args[0], '__call__')

    if is_func and (used_as_wrapper or expect_return_arg):
        func = args[0]
        sig_parameters = inspect.signature(func).parameters
        sig_argnames = [p.name for p in sig_parameters.values()]
        sig_defaults = {
            k: v.default
            for k, v in sig_parameters.items()
            if v.default is not inspect.Parameter.empty
        }

        @wraps(func)
        def _func(*args, **kwargs):
            if debug.disable:
                return func(*args, **kwargs)

            if debug._indent == 0:
                debug._stack = ""
            stack_before = debug._stack
            indent = ' ' * 4 * debug._indent
            debug._indent += 1

            args_kw = dict(zip(sig_argnames, args))
            defaults = {k: v for k, v in sig_defaults.items()
                        if k not in kwargs
                        if k not in args_kw}
            all_args = {**args_kw, **kwargs, **defaults}

            func_name = None
            if hasattr(func, '__name__'):
                func_name = func.__name__
            elif hasattr(func, '__class__'):
                func_name = func.__class__.__name__

            if func_name is None:
                func_name = '... ' + call_line + '...'
            else:
                func_name = '@' + func_name + '()'

            _debug_log('', indent=indent)
            _debug_log(func_name, indent=indent)

            debug._last_call = func
            debug._last_args = all_args
            debug._last_args_sig = sig_argnames

            for argtype, params in [("args", args_kw.items()),
                                    ("kwargs", kwargs.items()),
                                    ("defaults", defaults.items())]:
                if params:
                    _debug_log(f"{argtype}:", indent=indent + ' ' * 6)
                for argname, arg in params:
                    if argname == 'self':
                        # _debug_log(f"- self:  ...", indent=indent + ' ' * 8)
                        pass
                    else:
                        _debug_log(f"- {argname}:  ", arg, indent + ' ' * 8, assert_true)
            try:
                out = func(*args, **kwargs)
            except:
                _debug_crash_save()
                debug._stack = ""
                debug._indent = 0
                raise
            debug.out = out
            _debug_log("returned:  ", out, indent, assert_true)
            _debug_log('', indent=indent)
            debug._indent -= 1
            if not debug.full_stack:
                debug._stack = stack_before
            return out
        return _func
    else:
        if debug._indent == 0:
            debug._stack = ""
        argname = ')'.join('('.join(call_line.split('(')[1:]).split(')')[:-1])
        if assert_true:
            argname = ','.join(argname.split(',')[:-1])
            _debug_log(f"assert{{{argname}}}  ", args[0], ' ' * 4 * debug._indent, assert_true)
        else:
            for arg in args:
                _debug_log(f"{{{argname}}}  =  ", arg, ' ' * 4 * debug._indent, assert_true)
        if expect_return_arg:
            return args[0]
        return


def is_iterable(x):
    return isinstance(x, Iterable) or hasattr(x, '__getitem__') and not isinstance(x, str)


def ndarray_repr(t, assert_all=False):
    exception_encountered = False
    info = []
    shape = tuple(t.shape)
    single_entry = shape == () or shape == (1,)
    if single_entry:
        info.append(f"[{t.item():.4f}]")
    else:
        info.append(f"({', '.join(map(repr, shape))})")
    invalid_sum = (~np.isfinite(t)).sum().item()
    if invalid_sum:
        info.append(
            f"{invalid_sum} INVALID ENTR{'Y' if invalid_sum == 1 else 'IES'}")
        exception_encountered = True
    if debug.verbose > 1:
        if not invalid_sum and not single_entry:
            info.append(f"|x|={np.linalg.norm(t):.1f}")
            if t.size:
                info.append(f"x in [{t.min():.1f}, {t.max():.1f}]")
    if debug.verbose and t.dtype != np.float:
        info.append(f"dtype={str(t.dtype)}".replace("'", ''))
    if assert_all:
        assert_val = t.all()
        if not assert_val:
            exception_encountered = True
    if assert_all and not exception_encountered:
        output = "passed"
    else:
        if assert_all and not assert_val:
            output = f"ndarray({info[0]})"
        else:
            output = f"ndarray({', '.join(info)})"
    if exception_encountered and (not hasattr(debug, 'raise_exception') or debug.raise_exception):
        if debug.restore_defaults_on_exception:
            debug.raise_exception = False
            debug.silent = False
        debug.x = t
        msg = output
        debug._stack += output
        if debug._stack and '\n' in debug._stack:
            msg += '\nSTACK:  ' + debug._stack
        if assert_all:
            assert assert_val, "Assert did not pass on " + msg
        raise Exception("Invalid entries encountered in " + msg)
    return output


def tensor_repr(t, assert_all=False):
    exception_encountered = False
    info = []
    shape = tuple(t.shape)
    single_entry = shape == () or shape == (1,)
    if single_entry:
        info.append(f"[{t.item():.3f}]")
    else:
        info.append(f"({', '.join(map(repr, shape))})")
    invalid_sum = (~torch.isfinite(t)).sum().item()
    if invalid_sum:
        info.append(
            f"{invalid_sum} INVALID ENTR{'Y' if invalid_sum == 1 else 'IES'}")
        exception_encountered = True
    if debug.verbose and t.requires_grad:
        info.append('req_grad')
        if debug.verbose > 2:
            if t.is_leaf:
                info.append('leaf')
            if hasattr(t, 'retains_grad') and t.retains_grad:
                info.append('retains_grad')
    has_grad = (t.is_leaf or hasattr(t, 'retains_grad') and t.retains_grad) and t.grad is not None
    if has_grad:
        grad_invalid_sum = (~torch.isfinite(t.grad)).sum().item()
        if grad_invalid_sum:
            info.append(
                f"GRAD {grad_invalid_sum} INVALID ENTR{'Y' if grad_invalid_sum == 1 else 'IES'}")
            exception_encountered = True
    if debug.verbose > 1:
        if not invalid_sum and not single_entry:
            info.append(f"|x|={t.float().norm():.1f}")
            if t.numel():
                info.append(f"x in [{t.min():.2f}, {t.max():.2f}]")
        if has_grad and not grad_invalid_sum:
            if single_entry:
                info.append(f"grad={t.grad.float().item():.3f}")
            else:
                info.append(f"|grad|={t.grad.float().norm():.1f}")
    if debug.verbose and t.dtype != torch.float:
        info.append(f"dtype={str(t.dtype).split('.')[-1]}")
    if debug.verbose and t.device.type != 'cpu':
        info.append(f"device={t.device.type}")
    if assert_all:
        assert_val = t.all()
        if not assert_val:
            exception_encountered = True
    if assert_all and not exception_encountered:
        output = "passed"
    else:
        if assert_all and not assert_val:
            output = f"tensor({info[0]})"
        else:
            output = f"tensor({', '.join(info)})"
    if exception_encountered and (not hasattr(debug, 'raise_exception') or debug.raise_exception):
        if debug.restore_defaults_on_exception:
            debug.raise_exception = False
            debug.silent = False
        debug.x = t
        msg = output
        debug._stack += output
        if debug._stack and '\n' in debug._stack:
            msg += '\nSTACK:  ' + debug._stack
        if assert_all:
            assert assert_val, "Assert did not pass on " + msg
        raise Exception("Invalid entries encountered in " + msg)
    return output


def _debug_crash_save():
    if debug._indent:
        debug.args = debug._last_args
        debug.func = debug._last_call

        @wraps(debug.func)
        def _recall(*args, **kwargs):
            call_args = {**debug.args, **kwargs, **dict(zip(debug._last_args_sig, args))}
            return debug(debug.func)(**call_args)

        def print_stack(stack=debug._stack):
            print('\nSTACK:  ' + stack)
        debug.stack = print_stack

        debug.recall = _recall
    debug._indent = 0


def _debug_log(output, var=_NONE, indent='', assert_true=False, expand=True):
    debug._stack += indent + output
    if not debug.silent:
        print(indent + output, end='')
    if var is not _NONE:
        type_str = type(var).__name__.lower()
        if var is None:
            _debug_log('None')
        elif isinstance(var, str):
            _debug_log(f"'{var}'")
        elif type_str == 'ndarray':
            _debug_log(ndarray_repr(var, assert_true))
            if debug.show_tensor:
                _debug_show_print(var, indent=indent + 4 * ' ')
        # elif type_str in ('tensor', 'parameter'):
        elif type_str == 'tensor':
            _debug_log(tensor_repr(var, assert_true))
            if debug.show_tensor:
                _debug_show_print(var, indent=indent + 4 * ' ')
        elif hasattr(var, 'named_parameters'):
            _debug_log(type_str)
            params = list(var.named_parameters())
            _debug_log(f"{type_str}[{len(params)}] {{")
            for k, v in params:
                _debug_log(f"'{k}': ", v, indent + 6 * ' ')
            _debug_log(indent + 4 * ' ' + '}')
        elif is_iterable(var):
            expand = debug.expand_ignore != '*' and expand
            if expand:
                if isinstance(debug.expand_ignore, str):
                    if type_str == str(debug.expand_ignore).lower():
                        expand = False
                elif is_iterable(debug.expand_ignore):
                    for ignore in debug.expand_ignore:
                        if type_str == ignore.lower():
                            expand = False
            if hasattr(var, '__len__'):
                length = len(var)
            else:
                var = list(var)
                length = len(var)
            if expand and length > 0:
                _debug_log(f"{type_str}[{length}] {{")
                if isinstance(var, dict):
                    for k, v in var.items():
                        _debug_log(f"'{k}': ", v, indent + 6 * ' ', assert_true)
                else:
                    i = 0
                    for k, i in zip(var, range(debug.max_expand)):
                        _debug_log('- ', k, indent + 6 * ' ', assert_true)
                    if i < length - 1:
                        _debug_log('- ' + ' ' * 6 + '...', indent=indent + 6 * ' ')
                _debug_log(indent + 4 * ' ' + '}')
            else:
                _debug_log(f"{type_str}[{length}]")
        else:
            _debug_log(str(var))
    else:
        debug._stack += '\n'
        if not debug.silent:
            print()


def debug_show(x):
    assert is_iterable(x)
    debug(x)
    _debug_show_print(x, indent=' ' * 4 * debug._indent)


def _debug_show_print(x, indent=''):
    is_tensor = type(x).__name__ in ('Tensor', 'ndarray')
    if is_tensor:
        x = x.flatten()
    if type(x).__name__ == 'Tensor' and x.dim() == 0:
        return
    n_samples = min(10, len(x))
    di = len(x) // n_samples
    var = list(x[i * di] for i in range(n_samples))
    if is_tensor or type(var[0]) == float:
        var = [round(float(v), 4) for v in var]
    _debug_log('-->  ', str(var), indent, expand=False)


debug_init()
