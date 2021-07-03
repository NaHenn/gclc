#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
implementation of GCLC optimizer in Pytorch framework
"""
import torch
from torch.optim import Optimizer
from torch.optim.lbfgs import _strong_wolfe
from functools import reduce

from manip_scoptbfgs import _line_search_wolfe12, _LineSearchError

class GCLC(Optimizer):
    """Implements GCLC algorithm, heavily inspired by implementation of `L-BFGS
    <https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html#torch.optim.LBFGS>`_.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    .. note::
        This is a very memory intensive optimizer.

    Args:
        params (torch.Parameter): parameters to train
        opt (torch.Optimizer): optimizer for pre and post iterations
        lr (float): learning rate (default: 1)
        pre_iter (int): number of pre iterations (default: 3)
        post_iter (int): number of post iterations (default: 1)
        m (int): choice of restriction matrix (default: 0)
        l (int): choice of learning rate update (default: 1)
        line_search_fn (str): either 'strong_wolfe' or 'wolfe12' (default: 'wolfe12').
    """

    def __init__(self,
                 params,
                 opt,
                 lr=1,
                 pre_iter = 3,
                 post_iter = 1,
                 m = 0,
                 l = 1,
                 line_search_fn='strong_wolfe'):
        
        defaults = dict(
            lr=lr,
            pre_iter=pre_iter,
            post_iter=post_iter,
            m = m, 
            l = l,
            opt=opt,
            line_search_fn = line_search_fn)
        
        super(GCLC, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None
        
        if m >= self._numel():
            raise ValueError('Value for m is too large.')
        elif m < 0:
            raise ValueError('Value for m has to be larger or equal to 0.')
        if l >= self._numel():
            raise ValueError('Value for l is too large.')
        elif l < 0:
            raise ValueError('Value for l has to be larger or equal to 0.')

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)
    
    def _compute_Hessian(self, closure):   
        # TODO: this can be adapted to a sparse setting!
        loss = closure()
        grads = torch.autograd.grad(loss, self._params, retain_graph = True, create_graph = True)
        
        n_params = torch.cumsum(torch.tensor([0] + [int(p.numel()) for p in self._params[:-1]] + [1]),dim = 0)

        H = torch.zeros((self._numel(), self._numel()))
        for idx_param, p in enumerate(self._params):
            idx_col = 0
            for idx_grad, grad in enumerate(grads):
                grad = grad.flatten()
                for idx_g, g in enumerate(grad):
                    #print(idx_grad, idx_g, n_params[idx_grad] + idx_g, idx_col)
                    g2 = torch.autograd.grad(g, p, retain_graph = True, allow_unused=True)[0]
                    if g2 is None:
                        # print(torch.arange(n_params[idx_param].item(),n_params[idx_param+1].item()), n_params[idx_grad] + idx_g, ': None')                    
                        H[n_params[idx_param]:n_params[idx_param+1], idx_col] = 0
                    else:
                        H[n_params[idx_param]:n_params[idx_param+1], idx_col] = g2.flatten()
                    idx_col += 1
        return H
    
    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()
        
    def _clone_param_to_vector(self):
        # return torch.cat([p.clone(memory_format=torch.contiguous_format).flatten() for p in self._params])
        #return torch.cat([p.clone().flatten() for p in self._params])
        return torch.nn.utils.parameters_to_vector(self._params)
        
    def _set_param_from_vector(self, params_data):
        torch.nn.utils.vector_to_parameters(params_data, self._params)
        # offset = 0
        # for p in self._params:
        #     numel = p.numel()
        #     # view as to avoid deprecated pointwise semantics
        #     p = params_data[offset:offset + numel].view_as(p).clone()
        #     offset += numel
        # assert offset == self._numel()

    def _clone_param(self):
        #return [p.clone(memory_format=torch.contiguous_format) for p in self._params]
        return [p.clone().flatten() for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad
    
    # prepare line-search - note: these functions change the value of the parameters in the network
    def _net_loss(self, closure, x):
        #save_params = self._clone_param()
        self._set_param_from_vector(x)
        loss = closure()        
        #self._set_param_from_vector(save_params)       
        return loss.detach().item()        
        
    def _d_net_loss(self, closure, x):
        #save_params = self._clone_param()
        self._set_param_from_vector(x)
        loss = closure()  
        flat_grad = self._gather_flat_grad()
        #self._set_param_from_vector(save_params)   
        return flat_grad.detach().numpy()
    
    def _opt_iter(self, closure, opt, nr_iter):
        all_loss = []; all_grad = []; all_weights = []
        for i in range(nr_iter):
            all_weights.append(self._clone_param())
            loss = opt.step(closure)
            all_loss.append(float(loss))
            all_grad.append(self._gather_flat_grad())
            opt.zero_grad()        
        return all_loss, all_grad, all_weights

    @torch.no_grad()
    def step(self, closure, debug = False):
        """Performs a single optimization step.

        Args:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
            debug (bool): 
                If true return also the computed Hessian, gradients and weights at each step
                
        Returns:
            loss (float):
                loss value after one cycle
            error_code (int):
                0 - everything was fine
                1 - no lr update in Aggregation setting
                2 - eigenvalue decomposition was not possible, opt step instead
                3 - coarse problem could not be solved, opt step insteaad
                4 - line search did not converge, opt step instead
        """
        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = group['lr']
        pre_iter = group['pre_iter']
        post_iter = group['post_iter']
        opt=group['opt']
        m=group['m']
        l=group['l']
        line_search_fn = group['line_search_fn']

        # preoptimization
        pre_loss, pre_flat_grad, pre_weights = self._opt_iter(closure, opt, pre_iter)
        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault('n_cycles', 0)
        state.setdefault('func_eval', 0)


        # tensors cached in state (for tracing)
        if len(pre_flat_grad) > 1:
            prev_flat_grad = pre_flat_grad[-1]
            prev_loss = pre_loss[-1]
        else:
            prev_flat_grad = state.get('prev_flat_grad')
            prev_loss = state.get('prev_loss')
                    
        loss = closure()
        flat_grad = self._gather_flat_grad()
        loss = float(loss)
        pre_loss.append(loss)
        pre_flat_grad.append(flat_grad)
        
    

        t = lr
        ls_func_evals = state['func_eval']
        state['n_cycles'] += 1

        ############################################################
        # compute GCLC direction
        ############################################################
        error_code = 0
        # compute the Hessian
        with torch.enable_grad():
            H = self._compute_Hessian(closure).clone()
        # compute the restriction and prolongation operators
        if m == 0: # use aggregation
            P = torch.zeros((self._numel(),(self._numel()+1)//2)).index_put_((torch.arange(self._numel()),torch.arange(self._numel())//2),torch.ones(self._numel()))
            R = P.transpose(-2,-1)
        else: # default: use_smallest
            try:
                eig_vals, V = torch.symeig(H, eigenvectors = True)  
            except RuntimeError as error:
                print(error)
                error_code = 2 # failed SVD
                opt.step(closure)
                #flat_grad = self._gather_flat_grad()   # to save the direction
                opt.zero_grad()
                
            if error_code == 0: # default setting: use_smallest!
                P = V[:,:-m]
                R = P.transpose(-2,-1)
                
        # update the learning rate TODO: error_code
        if l > 0:
            if m == 0:
                try:
                    eig_vals,_ = torch.symeig(H, eigenvectors = False)  
                except RuntimeError as error:
                    print(error)
                    error_code = 1    # no new LR
                    new_lr = lr
            if l == 1:
                new_lr = 1/eig_vals[-1]
            else:
                new_lr = 2/(eig_vals[-1] + eig_vals[-l])
            #print(new_lr)
            group['lr'] = new_lr.item()
        
        # construct the CLC direction
        try:
            d = torch.matmul(P, torch.solve(torch.matmul(R, flat_grad).unsqueeze(1),
                                                                  torch.matmul(R,torch.matmul(H, P)))[0]).squeeze()
        except RuntimeError as error:
            print(error)
            error_code = 3 # system of equations can not be solved
            opt.step(closure)
            #flat_grad = self._gather_flat_grad()   # to save the direction
            opt.zero_grad()
        

        # if prev_flat_grad is None:
        #     prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
        # else:
        #     prev_flat_grad.copy_(flat_grad)
        # prev_loss = loss

        ############################################################
        # compute step length
        ############################################################
        # reset initial guess for step size
        if error_code < 2:    
            # directional derivative
            gtd = flat_grad.dot(-d)  # g * d
    
            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None:
                # perform line search, using user function
                if line_search_fn == "strong_wolfe":
                    x_init = self._clone_param()
    
                    def obj_func(x, t, d):
                        return self._directional_evaluate(closure, x, t, d)
    
                    loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                        obj_func, x_init, t, -d, loss, flat_grad, gtd)
                    print(t)
                    self._add_grad(-t, d)
                elif line_search_fn == "wolfe12":
                    x_init = self._clone_param_to_vector()
                    def f(x):
                        return self._net_loss(closure, x)
                    def fprime(x):
                        return self._d_net_loss(closure, x)
                    
                    if gtd < 0:
                        lsearch =  _line_search_wolfe12(f, fprime, x_init, -d, flat_grad, loss, prev_loss)
                        if lsearch[0] is None:
                            lsearch = _line_search_wolfe12(f, fprime, x_init, d, flat_grad, loss, prev_loss)
                            
                    else:
                        lsearch = _line_search_wolfe12(f, fprime, x_init, d, flat_grad, loss, prev_loss) 
                        if lsearch[0] is None:
                            lsearch = _line_search_wolfe12(f, fprime, x_init, -d, flat_grad, loss, prev_loss)
                    
                    t, func_evals, grad_evals, loss, prev_loss, flat_grad, ls_func_evals, tot_ge = lsearch
                     
                else: 
                    raise RuntimeError("only 'strong_wolfe' or 'wolfe12' is supported")
                if t is None:
                    error_code = 4 # line-search failed
                    opt.step(closure)
                    #flat_grad = self._gather_flat_grad()   # to save the direction                    
                    opt.zero_grad()    
                    loss = closure().item()
                    flat_grad = self._gather_flat_grad()
                    
            else:
                print('fixed{}'.format(t))
                # no line search, simply move with fixed-step
                self._add_grad(t, -d)
                loss = float(closure())
                flat_grad = self._gather_flat_grad()
        else:
            loss = closure()
            flat_grad = self._gather_flat_grad()
        
        post_loss, post_flat_grad, post_weights = self._opt_iter(closure, opt, post_iter)
        if len(post_loss) > 0:
            loss = post_loss[-1]
            flat_grad = post_flat_grad[-1]
        post_loss.append(closure().item())
        post_flat_grad.append(self._gather_flat_grad())

        
        state['prev_flat_grad'] = flat_grad
        state['prev_loss'] = loss
        state['all_loss'] = pre_loss + post_loss   
        state['error_code'] = error_code
        state['func_eval'] = ls_func_evals
        if debug:
            state['all_grad'] = pre_flat_grad + post_flat_grad
            state['H'] = H.clone()
            state['all_weights'] = pre_weights + post_weights
            state['d'] = d
            state['t'] = t
            state['x_init'] = x_init
            
        return loss, error_code