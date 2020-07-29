import scipy.misc
import scipy.special
import scipy.stats
import scipy
import numpy as np


def dbl_epsilon(_eps=[]):
    if not _eps:
        etop = 1.0
        ebot = 0.0
        eps = ebot+(etop-ebot)/2.0
    while eps!=ebot and eps!=etop:
        epsp1 = 1.0 - eps
        if epsp1<1.0: etop = eps
        else: ebot = eps
        eps = ebot+(etop-ebot)/2.0
        _eps.append(etop)
    assert (1.0-etop)<1.0 and (1.0-ebot)==1.0, 'Error in epsilon calculation'
    return _eps[10]

DBL_EPSILON = dbl_epsilon()

def my_log_ndtr(a):
    print 'my_log_ndtr', a

    if (a > 6):
        print 'a > 6'
        print -scipy.special.ndtr(-a)
        return -scipy.special.ndtr(-a)
    
    if (a > -20):
        print 'a > -20'
        print scipy.special.ndtr(a)
        print np.log(scipy.special.ndtr(a))
        return np.log(scipy.special.ndtr(a))
    
    last_total = 0
    right_hand_side = 1
    numerator = 1
    denom_factor = 1
    denom_cons = 1.0 / (a * a);
    sign = 1
    i = 0
    
    
    log_LHS = -0.5 * a * a - np.log(-a) - 0.5 * np.log(2 * np.pi)
    print np.log(-a)
    print 'log_LHS', log_LHS
    
    while (np.fabs(last_total - right_hand_side) > DBL_EPSILON):
        i += 1
        last_total = right_hand_side
        sign = -sign
        denom_factor *= denom_cons
        numerator *= 2 * i - 1
        #print numerator
        #print denom_factor
        #print numerator * denom_factor
        right_hand_side += (
                            sign * 
                            numerator * 
                            denom_factor)

    
    return log_LHS + np.log(right_hand_side)

logndtr_fn = np.vectorize(my_log_ndtr)
logndtr_fn = scipy.special.log_ndtr

_norm_pdf_C = np.sqrt(2*np.pi)
_norm_pdf_logC = np.log(_norm_pdf_C)


def _norm_pdf(x):
    return np.exp(-x**2/2.0) / _norm_pdf_C


def _norm_logpdf(x):
    return -x**2 / 2.0 - _norm_pdf_logC

class truncnorm2_gen(scipy.stats.rv_continuous):
    """A truncated normal continuous random variable.
    %(before_notes)s
    Notes
    -----
    The standard form of this distribution is a standard normal truncated to
    the range [a, b] --- notice that a and b are defined over the domain of the
    standard normal.  To convert clip values for a specific mean and standard
    deviation, use::
        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    `truncnorm` takes ``a`` and ``b`` as shape parameters.
    %(after_notes)s
    %(example)s
    """
    def _argcheck(self, a, b):
        self.a = a
        self.b = b
        self._nb = scipy.special.ndtr(b)
        self._na = scipy.special.ndtr(a)
        self._lna =logndtr_fn(a)
        self._lnb =logndtr_fn(b)
        self._sb = scipy.special.ndtr(-b)
        self._sa = scipy.special.ndtr(-a)
        self._delta = np.where(self.a > 0,
                               -(self._sb - self._sa),
                               self._nb - self._na)
        self._logdelta = np.log(self._delta)
        self._logdelta2 = scipy.misc.logsumexp(a= [self._lnb, self._lna], b=[1, -1])
        
        #print 'logdelta', self._logdelta
        #print 'logdelta2', self._logdelta2
        return a != b

    def _pdf(self, x, a, b):
        return _norm_pdf(x) / self._delta

    def _logpdf(self, x, a, b):
        return _norm_logpdf(x) - self._logdelta

    def _cdf(self, x, a, b):
        
        return (scipy.special.ndtr(x) - self._na) / self._delta

    def _logcdf(self, x, a, b):
        #print '_logcdf'
        
        #print 'logndtr_fn(x)', logndtr_fn(x)
        t = np.vstack([[logndtr_fn(x), np.ones(x.shape)*self._lna]]).transpose()
        #print 't', t
        
        #print 'logsumexp', scipy.misc.logsumexp(a= t, axis=1, b=[1, -1])
        
        
        return scipy.misc.logsumexp(a= t, axis=1, b=[1, -1]) - self._logdelta2
    

        
    def _logsf(self, x, a, b):
        #logcdf = scipy.stats.rv_continuous._logcdf(self, x, a, b)
        #print x.shape
        logcdf = self._logcdf(x, a, b)
        #print 'logcdf', logcdf
        t = np.vstack([[np.zeros(logcdf.shape), logcdf]]).transpose()
        
        return scipy.misc.logsumexp(a= t, axis=1, b=[1, -1])

        
    def _ppf(self, q, a, b):
        # XXX Use _lazywhere...
        ppf = np.where(self.a > 0,
                       _norm_isf(q*self._sb + self._sa*(1.0-q)),
                       _norm_ppf(q*self._nb + self._na*(1.0-q)))
        return ppf

    def _stats(self, a, b):
        nA, nB = self._na, self._nb
        d = nB - nA
        pA, pB = _norm_pdf(a), _norm_pdf(b)
        mu = (pA - pB) / d   # correction sign
        mu2 = 1 + (a*pA - b*pB) / d - mu*mu
        return mu, mu2, None, None
truncnorm2 = truncnorm2_gen(name='truncnorm2')





## scale graphs 
#
#a = -2001.
#b = 2001
#loc = 0.
#scale = 5.
#
#a_scaled, b_scaled = (a - loc) / scale, (b - loc) / scale
#print a, b
#rv1 = scipy.stats.truncnorm(a_scaled, b_scaled, loc=loc, scale=scale)
#rv2 = truncnorm2(a_scaled, b_scaled, loc=loc, scale=scale)
#
#print rv1.logsf(2000)
#print rv2.logsf(2000)
#
#print rv1.logcdf(-2000)
#print rv2.logcdf(-2000)
#
#dist_space = np.linspace( a, b, 100000 )
#
#
## plot the results
#import matplotlib.pyplot as plt
#f, axarr = plt.subplots(2, sharex=True)
#axarr[0].set_ylabel('old')
#axarr[0].plot( dist_space, rv1.logsf(dist_space) )
#axarr[1].set_ylabel('new')
#axarr[1].plot( dist_space, rv2.logsf(dist_space) )
#plt.title('difference in log survival function')
#plt.savefig("difference in log survival function.png")
#
#
#f, axarr = plt.subplots(2, sharex=True)
#axarr[0].set_ylabel('old')
#axarr[0].plot( dist_space, rv1.cdf(dist_space) )
#axarr[1].set_ylabel('new')
#axarr[1].plot( dist_space, rv2.cdf(dist_space) )
#plt.title('difference in cdf')
#plt.savefig("difference in cdf.png")
#
#
#f, axarr = plt.subplots(2, sharex=True)
#axarr[0].set_ylabel('old')
#axarr[0].plot( dist_space, rv1.logcdf(dist_space) )
#axarr[1].set_ylabel('new')
#axarr[1].plot( dist_space, rv2.logcdf(dist_space) )
#plt.title('difference in log cdf')
#plt.savefig("difference in log cdf.png")






np.set_printoptions(precision=40)

# floating point graph

a = 0.
b = 20.
loc = 0.
scale = 2.

dist_space = np.power(0.1, np.arange(30), dtype=np.double)

a_scaled, b_scaled = (a - loc) / scale, (b - loc) / scale
print a, b
print a_scaled, b_scaled
print dist_space
print dist_space.shape
print (dist_space-loc)*1.0/scale
rv1 = scipy.stats.truncnorm(a_scaled, b_scaled, loc=loc, scale=scale)
rv2 = truncnorm2(a_scaled, b_scaled, loc=loc, scale=scale)

#print np.log(rv2.cdf(dist_space))
#print rv2.logcdf(dist_space)


# plot the results
print rv1.logsf(dist_space) 
print rv2.logsf(dist_space) 
#print rv2.parentlogsf(dist_space)

#factor = 5.
#
#a_scaled, b_scaled = (a - loc) / (scale*factor), (b - loc) / (scale*factor)
#print a_scaled, b_scaled
#rv3 = truncnorm2(a_scaled, b_scaled, loc=loc, scale=scale*factor)
#dist_space_2 = ((dist_space - loc) * factor) + loc
#print dist_space_2
#print rv3.logsf(dist_space_2) 




#import matplotlib.pyplot as plt
#f, axarr = plt.subplots(2, sharex=True)
#axarr[0].set_ylabel('old')
#axarr[0].plot( dist_space, rv1.logsf(dist_space) )
#axarr[1].set_ylabel('new')
#axarr[1].plot( dist_space, rv2.logsf(dist_space) )
#plt.title('floating point test log survival function')
#plt.savefig("floating point test log survival function.png")