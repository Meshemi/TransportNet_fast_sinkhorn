import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import os
np.set_printoptions(suppress=True)

change_mode = lambda v: -v -1/2
mode_article = change_mode #switch to the lambda_l as it is used in article
mode_code = change_mode #switch to the lambda_l as it is used in default code

class Sinkhorn:

    def __init__(self, L, W, people_num, iter_num, eps, L0 = 1, max_inner_step = 100, cut_L_f = lambda L: L/2, rise_L_f = lambda L: L*2, 
        log_file = None, verbose = True):
        #cut_L_f - function of L reducing value of L at each step of sinkhorm, None if skip 
        #rise_L_f - function of L rising L at step when out condition of sinkhorn failes, None if ignore condition
        #log_file - path to ...log.txt for save testing and debbuging info
        #verbose - set false to print log only to file
        #max_inner_step - stop inner sinkhorn step after some amount of failed out conditions
        #L0 - initial estimation of Lip. constant
        #initialize without new parameters to use it like default sinkhorn or like a article-like fast sinkhorn

        #To run fast sinkhorn use self.iterate(cost_matrix, mode = 'fast')

        self.L = L
        self.W = W
        assert (len(L) == len(W))
        self.n = len(L)
        self.people_num = people_num
        self.num_iter = iter_num
        self.eps = eps
        self.multistage_i = 0

        self.cut_L_f = cut_L_f if cut_L_f is not None else lambda L: L
        self.rise_L_f = rise_L_f if rise_L_f is not None else lambda L: None
        self.L0 = L0
        self.max_inner_step = max_inner_step
        self.set_log(log_file) #better use logging here
        self.verbose = verbose

        self.history = {'idx': [], 'L': [], 'phi':[None]} #just for debugging
        

    def initialize(self):
        self.previous_params = {'L': self.L0, 'a': 0, 'v': np.zeros((2, self.n))}
        self.temp_params = {}

    def update_params(self):
        self.previous_params.update(self.temp_params)

    def set_log(self, log_file):
        self.log_file = log_file
        if log_file is not None:
            os.makedirs(os.path.dirname(log_file), exist_ok = True)

    def log(self, *args, verbose = True):
        if self.log_file is not None:
            with open(self.log_file, 'a') as f:
                print(*args, file = f)
        if verbose and self.verbose:
            print(*args)


    def default_sinkhorn(self, k, cost_matrix, lambda_W_prev, lambda_L_prev):
        self.inner_step += 1
        if k % 2 == 0:
            lambda_W = lambda_W_prev
            lambda_L = np.log(np.nansum(
                (np.exp(-lambda_W_prev - 1 - cost_matrix)).T
                / self.L, axis=0
            ))
        else:
            lambda_L = lambda_L_prev
            lambda_W = np.log(np.nansum(
                (np.exp(-lambda_L - 1 - cost_matrix.T)).T
                / self.W, axis=0
            ))
        return lambda_W, lambda_L

    def iterate(self, cost_matrix, mode = 'fast'):
        #run with mode default to use default sinkhorn
        #run with mode fast to use fast sinkhorn
        assert(mode in ['fast', 'default'])

        cost_matrix[cost_matrix == 0.0] = 100.0

        lambda_L = np.zeros(self.n)
        lambda_W = np.zeros(self.n)

        if mode == 'fast':
            self.initialize()
        self.inner_step = 0
        for k in range(self.num_iter):

            if mode == 'default':
                lambda_Wn, lambda_Ln = self.default_sinkhorn(k, cost_matrix, lambda_W, lambda_L)
                out_f = lambda x: x
            else:
                lambda_Wn, lambda_Ln = self.sinkhorn(k, cost_matrix, lambda_W, lambda_L)
                out_f = mode_code

            delta = np.linalg.norm(np.concatenate((lambda_Ln - lambda_L,
                                                   lambda_Wn - lambda_W)))

            lambda_L, lambda_W = lambda_Ln, lambda_Wn

            if delta < self.eps:
                self.log(f"number of iterations in Sinkhorn:{self.inner_step}")
                self.log('L evolution: [{}]'.format(','.join(list(map(lambda x: str(x), self.history['L'])))))
                if self.log_file is not None:
                    show = self.history['phi'][-1]
                    if show is None:
                        show = self.phi(lambda_L, lambda_W, cost_matrix = cost_matrix)
                    self.log('Phi: {}'.format(show))
                break
        lambda_L, lambda_W = out_f(lambda_L), out_f(lambda_W)
        r = self.rec_d_i_j(lambda_L, lambda_W, cost_matrix)
        return r, lambda_L, lambda_W

    def rec_d_i_j(self, lambda_L, lambda_W, cost_matrix):
        er = np.exp(-1 - cost_matrix - (np.reshape(lambda_L, (self.n, 1)) + lambda_W))
        return er * self.people_num


    def get_B(self, B = None, **kwargs):
        if B is not None:
            return B
        lambda_L = kwargs['lambda_L']
        lambda_W = kwargs['lambda_W']
        cost_matrix = kwargs['cost_matrix']
        return np.exp(lambda_L[:, None] + lambda_W[None, :] - cost_matrix)

    def grad_phi(self, right_B_sum = None, left_B_sum = None, B_sum = None, **kwargs):
        returns = []
        r = right_B_sum
        l = left_B_sum
        S = B_sum
        if r is None or l is None:
            B = self.get_B(**kwargs)
        if r is None:
            r = np.nansum(B, axis = 1)
        if l is None:
            l = np.nansum(B, axis = 0)
        if S is None:
            S = np.nansum(r)
            returns = [S]
        g = [-self.L + r/S, -self.W + l/S]
        return tuple([np.array(g)] + returns)

    def phi(self, lambda_L, lambda_W, B_sum = None, **kwargs):
        returns = []
        if B_sum is None:
            B = self.get_B(lambda_L = lambda_L, lambda_W = lambda_W, **kwargs)
            B_sum = np.nansum(B)
            # returns = [B_sum]
        a1 = np.log(B_sum)
        a2 = lambda_L @ self.L
        a3 = lambda_W @ self.W
        res = a1 - a2 - a3
        # return tuple([res] + returns)
        return res


    def sinkhorn(self, k, cost_matrix, lambda_W_prev, lambda_L_prev):
        
        self.temp_params['L'] = self.cut_L_f(self.previous_params['L'])
        k = 0
        while True:

            k += 1
            L = self.temp_params['L']
            self.history['L'].append(L)
            assert(L > 0)
            m1 = 1/L

            self.temp_params['a'] = m1*(0.5 + (0.25 + self.previous_params['a']**2*self.previous_params['L']*L)**0.5)
            a = self.temp_params['a']
            tau = 1/(a*L)

            xk = np.array([lambda_L_prev, lambda_W_prev])
            y = tau*self.previous_params['v'] + (1 - tau)*xk # v y and x is equal at first stage
            g, S = self.grad_phi(lambda_L = y[0], lambda_W = y[1], cost_matrix = cost_matrix)

            grad_norm = np.linalg.norm(g, axis = 1)**2
            
            i = np.argmax(grad_norm)

            lambda_W, lambda_L = self.default_sinkhorn(i, cost_matrix, *list(map(mode_code, xk[::-1])))
            lambda_W, lambda_L = mode_article(lambda_W), mode_article(lambda_L)
            self.temp_params['v'] = self.previous_params['v'] - a*g
            
            new_L = self.rise_L_f(self.temp_params['L'])
            phi_new = None
            if new_L is None:
                break
            phi_new = self.phi(lambda_L, lambda_W, cost_matrix = cost_matrix)
            phi_temp = self.phi(*y, B_sum = S)
            # print('Check delta: ', np.linalg.norm(g) - np.sum(grad_norm))
            to_compare = phi_temp - np.sum(grad_norm)/(2*L)
            
            if phi_new <= to_compare or k >= self.max_inner_step:
                break
            self.temp_params['L'] =  new_L
        self.history['phi'].append(phi_new)
        self.history['idx'].append(i) 
        self.update_params()
        return lambda_W, lambda_L
