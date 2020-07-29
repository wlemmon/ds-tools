import numpy as np

# some constants
twoPi = 2 * np.pi
measurement_noise = 5.
sigma_sq_times_2 = (measurement_noise ** 2)*2
log_sqrt_twoPi_sigmaSq = np.log(np.sqrt(twoPi*sigma_sq_times_2/2))

""" 
lognormpdf_fast
Computes the log of a noral distribution, omitting the normalization element.
This is useful for computing likelihoods over a set of gaussian models with identical sigmas.
Essentially, it just computes the exponent of a gaussian.
Compatible with numpy.

Example:

print lognormpdf_fast(np.random.randn(5), np.random.randn(5), np.random.randn(5), np.random.randn(5), sigma_sq_times_2, log_sqrt_twoPi_sigmaSq)

>> [-2.62435161 -2.58529663 -2.65249398 -2.53742785 -2.55496317]

"""
def lognormpdf_fast(x1, y1, x2, y2, sigma_sq_times_2, log_sqrt_twoPi_sigmaSq):
    distSq = (x2 - x1) ** 2 + (y2 - y1) ** 2
    return - distSq / sigma_sq_times_2 - log_sqrt_twoPi_sigmaSq

"""
normpdf_fast
Same as above, but not in log space.
"""
def normpdf_fast(x1, y1, x2, y2, sigma_sq_times_2, log_sqrt_twoPi_sigmaSq):
    return np.exp(lognormpdf_fast(x1, y1, x2, y2, sigma_sq_times_2, log_sqrt_twoPi_sigmaSq))

# just here for comparison
def lognormpdf(x, mu, sigma):
    return - (((mu - x) ** 2) / (sigma ** 2) / 2.0) - np.log(np.sqrt(twoPi * (sigma ** 2)))

# just here for comparison
def normpdf(x, mu, sigma):
    return np.exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / np.sqrt(twoPi * (sigma ** 2))
    
# in case you want to normalize a list of log weights.
def lognormalize(x):
    a = np.logaddexp.reduce(x)
    return np.exp(x - a)
if __name__ == '__main__':
    assert np.allclose(lognormpdf(1., 2., 3.), np.log( normpdf(1., 2., 3.)))
    assert np.allclose(lognormpdf_fast(0., 0., 0., 0., sigma_sq_times_2, log_sqrt_twoPi_sigmaSq), np.log(normpdf_fast(0., 0., 0., 0., sigma_sq_times_2, log_sqrt_twoPi_sigmaSq)))
    print lognormpdf_fast(np.random.randn(5), np.random.randn(5), np.random.randn(5), np.random.randn(5), sigma_sq_times_2, log_sqrt_twoPi_sigmaSq)