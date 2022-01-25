import scipy

import logging, sys
import numpy as np
import scipy.stats
import autograd
import autograd.numpy as anp
import autograd.scipy as ascipy
from autograd.builtins import isinstance, list, dict, tuple  # TODO: reconsider to use this line

import util
from expr import *


########
# init #
########

def init(e): pass

#############
# auxiliary #
#############

def merge_fs(fs, r0, g):
    """
    Args:    
      - fs : (a -> c) list 
      - r0 : c
      - g  : c * c -> c
    Returns: 
      - f  : a -> c
    where
      f(x) = fold_left g r0 [fs[0](x); fs[1](x); ...; fs[n-1](x)]
    """
    n = len(fs)
    def _f(_x, n=n, fs=fs, r0=r0, g=g):
        res = r0
        for i in range(n):
            res = g(res, fs[i](_x))
        return res
    return _f

def merge_logpqs(logpqs):
    """
    Args:    logpqs : logpq list
    Returns: lambda _thts: \sum_i logpqs[i](_thts)
    """
    def g(z0,z1): return z0+z1
    return merge_fs(logpqs, 0, g)

###############
# eval_smooth #
###############

def eval_smooth(eta, e, thts, env={}):
    """
    Summary: computes the smoothed gradient estimator.
    Args:
      - eta   : float
      - e     : Expr
      - thts  : float array
      - env   : (str -> (func * float)) dict
    Returns:
      - ret   : func
      - retvl : float 
      - epss  : float array (not accurate)
      - xs    : float array (not accurate)
      - logpq : func
    where
      - eta = smoothing coefficient
      - env[var_str] = return value of Var(var_str) as (function of \THT, float)
      - ret(\THT) = return value of e (as a function of \THT)
      - retvl  = ret(thts)
      - epss   = values sampled from N(0,1)
      - xs     = T_thts(epss)
      - logpq(\THT) = log p(X,Y) - log q_\THT(X) |_{X=T_\THT(epss)}
    here capital math symbols denote vectors.
    """
    
    if isinstance(e, Cnst):
        ret   = lambda _thts, c=e.c: c
        retvl = ret([])
        epss  = anp.array([])
        xs    = anp.array([])
        logpq = lambda _thts: 0.0

    elif isinstance(e, Var):
        assert(e.v in env)
        (ret, retvl) = env[e.v]
        epss  = anp.array([])
        xs    = anp.array([])
        logpq = lambda _thts: 0.0

    elif isinstance(e, Linear):
        ret   = None # ASSUME: (Linear ...) appear only in the conditional part of If.
        retvl = e.c0 + sum([ci*env[vi][1] for (ci,vi) in e.cv_l])
        epss  = anp.array([])
        xs    = anp.array([])
        logpq = lambda _thts: 0.0
        
    elif isinstance(e, App):
        # recursive calls
        num_args = len(e.args)
        (ret_sub, retvl_sub, epss_sub, xs_sub, logpq_sub)\
            = zip(*[ eval_smooth(eta, e.args[i], thts, env) for i in range(num_args) ])

        # compute: all but ret, retvl
        epss  = anp.concatenate( epss_sub)
        xs    = anp.concatenate(   xs_sub)
        logpq = merge_logpqs   (logpq_sub)
        
        # compute: ret, retvl
        op = App.OP_DICT[num_args][e.op]
        ret   = lambda _thts, op=op, ret_sub=ret_sub, num_args=num_args:\
                op(*[  ret_sub[i](_thts) for i in range(num_args)])
        retvl = op(*[retvl_sub[i]        for i in range(num_args)])
            
    elif isinstance(e, If):
        # recursive calls
        (_, retvl_1, epss_1, xs_1, logpq_1)\
            =  eval_smooth(eta, e.e1, thts, env)

        (ret_t, retvl_t, epss_t, xs_t, logpq_t) = eval_smooth(eta, e.e2, thts, env)
        (ret_f, retvl_f, epss_f, xs_f, logpq_f) = eval_smooth(eta, e.e3, thts, env)

        c = ascipy.special.expit(retvl_1 / eta)

        def logpq_r(thts, c=c, logpq_t=logpq_t, logpq_f=logpq_f):
          return c * logpq_t(thts) + (1 - c) * logpq_f(thts)
        
        # compute: all
        ret = lambda thts, c=c, ret_t=ret_t, ret_f=ret_f:\
                  c * ret_t(thts) + (1 - c) * ret_f(thts)
        retvl = c * retvl_t + (1 - c) * retvl_f
        logpq = merge_logpqs   ([logpq_1, logpq_r])

        # Not really accurate but are ignored anyway
        epss  = anp.concatenate(( epss_1, epss_t, epss_f ))
        xs    = anp.concatenate((   xs_1,   xs_t, xs_f ))  
            
    elif isinstance(e, Let):
        # recursive calls
        (ret_1, retvl_1, epss_1, xs_1, logpq_1) = eval_smooth(eta, e.e1, thts, env)
        env_new = util.copy_add_dict(env, {e.v1.v : (ret_1, retvl_1)})
        (ret_2, retvl_2, epss_2, xs_2, logpq_2) = eval_smooth(eta, e.e2, thts, env_new)
        
        # compute: all
        ret   = ret_2
        retvl = retvl_2
        epss  = anp.concatenate(( epss_1, epss_2 ))
        xs    = anp.concatenate((   xs_1,   xs_2 ))
        logpq = merge_logpqs   ([logpq_1, logpq_2])

    elif isinstance(e, Sample):
        # recursive calls
        (ret_1, retvl_1, epss_1, xs_1, logpq_1) = eval_smooth(eta, e.e1, thts, env)
        (ret_2, retvl_2, epss_2, xs_2, logpq_2) = eval_smooth(eta, e.e2, thts, env)

        # compute: all but logpq
        stind = e.stind['thts']
        eps_3     = np.random.normal(0, 1) # do sampling
        eps2x_cur = lambda _tht, eps=eps_3: _tht[0] + util.softplus_anp(_tht[1]) * eps
        eps2x_3   = lambda _thts, eps2x_cur=eps2x_cur, stind=stind: eps2x_cur(_thts[stind:stind+2])
        x_3       = eps2x_3(thts)

        ret   = lambda _thts, eps2x_3=eps2x_3: eps2x_3(_thts)
        retvl = x_3  # use current thts value to compute return value
        epss  = anp.concatenate(( epss_1, epss_2, anp.array([eps_3]) ))
        xs    = anp.concatenate((   xs_1,   xs_2, anp.array([  x_3]) ))

        # compute: logpq
        def logpq_3(_thts, ret=ret, ret_1=ret_1, ret_2=ret_2, stind=stind):
            # compute: log p(x|p_loc,p_scale) - log q(x|q_loc,q_scale)
            x       = ret  (_thts)
            p_loc   = ret_1(_thts)
            p_scale = ret_2(_thts)
            q_loc   =                   _thts[stind]
            q_scale = util.softplus_anp(_thts[stind+1])
            return (ascipy.stats.norm.logpdf(x, p_loc, p_scale) -\
                    ascipy.stats.norm.logpdf(x, q_loc, q_scale))
        
        logpq = merge_logpqs([logpq_1, logpq_2, logpq_3])

    elif isinstance(e, Fsample):
        # recursive calls
        (ret_1, retvl_1, epss_1, xs_1, logpq_1) = eval_smooth(eta, e.e1, thts, env)
        (ret_2, retvl_2, epss_2, xs_2, logpq_2) = eval_smooth(eta, e.e2, thts, env)

        # compute: all
        x_3 = np.random.normal(retvl_1, retvl_2)  # do sampling
        ret   = lambda _thts, x_3=x_3: x_3
        retvl = x_3
        epss  = anp.concatenate(( epss_1, epss_2 ))
        xs    = anp.concatenate((   xs_1,   xs_2 ))
        logpq = merge_logpqs   ([logpq_1, logpq_2])

    elif isinstance(e, Observe): 
        # recursive calls
        num_args = len(e.args)
        (ret_sub, retvl_sub, epss_sub, xs_sub, logpq_sub)\
            = zip(*[ eval_smooth(eta, e.args[i], thts, env) for i in range(num_args) ])

        # compute: all but logpq
        ret   = lambda _thts, c=e.c1.c: c
        retvl = ret([])
        epss  = anp.concatenate( epss_sub)
        xs    = anp.concatenate(   xs_sub)

        # compute: logpq
        dstr_logpdf = Observe.DSTR_DICT[e.dstr]
        def logpq_cur(_thts, dstr_logpdf=dstr_logpdf, c=e.c1.c,
                      ret_sub=ret_sub, num_args=num_args):
            # compute: log p(c|p_loc,p_scale)
            return dstr_logpdf(c, *[ret_sub[i](_thts) for i in range(num_args)])

        logpq = merge_logpqs(list(logpq_sub) + [logpq_cur])

    else: assert(False)
    return (ret, retvl, epss, xs, logpq)

#############
# elbo_grad #
#############

def elbo_grad(e, thts, idx):
    assert(isinstance(e, Expr))

    coefficient = 10
    eta = coefficient * (idx + 1) ** (-0.5)

    (_, _, _, _, logpq_fun) = eval_smooth(eta, e, thts)
    res = autograd.grad(logpq_fun)(thts)

    return res
