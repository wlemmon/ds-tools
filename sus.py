from __future__ import print_function
import numpy as np
import random
logOfTwo = np.log(2)

"""
stochastic_universal_sampling_logs
Resamples s_prev by relative LOG weights
s_prev can be states, indices, etc.
w_prev are weights that do NOT have to be normalized
n is the desired number of samples, or wagon spokes.
Example:
print stochastic_universal_sampling_logs(range(5), np.log(np.arange(5))+1, 10)
>> ([0, 1, 2, 2, 3, 3, 3, 4, 4, 4], [0.0, 0.69314718055994529, 1.0986122886681098, 1.0986122886681098, 1.3862943611198906, 1.3862943611198906, 1.3862943611198906, 1.6094379124341003, 1.6094379124341003, 1.6094379124341003])
"""
def stochastic_universal_sampling_logs(s_prev, w_prev, n):
    s = []
    w = []
   
    # very wows. computes a running sum in log space
    c = np.logaddexp.accumulate(w_prev)
    
    # divide by n in log space
    nInvLog = c[-1] - np.log(n)
    
    #u_j = np.log(random.uniform(0, nInv))
    # pick a 'random' point between log(0) and nInvLog; since I cannot figure out how to do this, just divide by two in log space.
    u_j = nInvLog - logOfTwo
    i = 0
    for j in range(n):
        while u_j > c[i]:
            i += 1
        s.append(s_prev[i])
        w.append(w_prev[i])
        u_j = np.logaddexp(u_j, nInvLog)
    return s, w

    
# just here for comparison with the log version
def stochastic_universal_sampling(s_prev, w_prev, n):
    #assert len(s_prev) == len(w_prev)
    s = []
    w = []
    
    c = np.cumsum(w_prev,axis=0)

    nInv = c[-1]/n
    u_j = random.uniform(0, nInv)
    i = 0
    
    for j in range(n):
        #assert i < len(c)
        # a while loop is faster than searchsorted in most cases. the only time this is false is when n << len(w) AND the standard deviation of the data is very small. In that case, we would expect to jump over large numbers of weights. But this never happens in practice.
        while u_j > c[i]:
            i += 1
        s.append(s_prev[i])
        w.append(w_prev[i])
        u_j += nInv
    return s, w

# slow cumulative log sum. only here for testing
def logcumsum(w):

    c = [w[0]]
    for a in w[1:]:
        c.append(np.logaddexp(c[-1], a))
    return c
    
if __name__ == '__main__':
    assert stochastic_universal_sampling_logs(['a', 'b', 'c', 'd', 'e'], map(lambda x : np.log(x), [0.2]*5), 10)[0] == ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'e', 'e']
    assert stochastic_universal_sampling_logs(['a', 'b'], map(lambda x : np.log(x), [0.0001, 0.9999]), 10)[0] == ['b']*10
    assert stochastic_universal_sampling_logs(['a', 'b'], map(lambda x : np.log(x), [0.9999, 0.0001]), 10)[0] == ['a']*10
    assert stochastic_universal_sampling_logs(range(5), map(lambda x : np.log(x), [0.9999, 0.0001]), 10)[0] == [0]*10
    assert stochastic_universal_sampling(['a', 'b'], [0.1, 0.1], 4)[0] == ['a', 'a', 'b', 'b']
    assert stochastic_universal_sampling(['a', 'b', 'c'], [1, 1, 1000], 3)[0] == ['c']*3
    print(stochastic_universal_sampling_logs(range(5), np.log(np.arange(5)+1), 10))
    
        
    assert np.allclose(np.logaddexp.accumulate([1, 2, 3]), logcumsum([1, 2, 3]))
    import timeit
    print(timeit.timeit('logcumsum(w)', setup="import numpy as np; from sus import logcumsum; w=range(1000)", number = 1000))
    print(timeit.timeit('logcumsum(w)', setup="import numpy as np; from sus import logcumsum; w=np.arange(1000)", number = 1000))
    print(timeit.timeit('np.logaddexp.accumulate(w)', setup="import numpy as np; from sus import logcumsum; w=range(1000)", number = 1000))
    print(timeit.timeit('np.logaddexp.accumulate(w)', setup="import numpy as np; from sus import logcumsum; w=np.arange(1000)", number = 1000))