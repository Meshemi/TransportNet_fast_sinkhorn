# TransportNet_fast_sinkhorn

File sinkhorn.py contain code of fast sinkhorn for TransportNet project. Initialize class as a default one:
s = Sinkhorn(L, W, people_num, sink_num_iter, sink_eps)
Then you can use 'rec, _, _ = s.iterate(cost_matrix, mode = mode)' with mode in ['fast', 'default'] to correspondingly use fast or default sinkhorn implemetation.
Some additional parameters of Sinkhorn class:
  
        cut_L_f - function of L reducing value of L at each step of sinkhorm, None if skip 
        rise_L_f - function of L rising L at step when out condition of sinkhorn failes, None if ignore condition
        log_file - path to ...log.txt for save testing and debbuging info
        verbose - set false to print log only to file
        max_inner_step - stop inner sinkhorn step after some amount of failed out conditions
        L0 - initial estimation of Lip. constant
