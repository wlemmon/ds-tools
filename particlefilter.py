from __future__ import print_function
from print_indentation_utils import *
from sus import *
import numpy as np
from math import *
from profiling import *
from norm import *
from coordinateMappings import *

twoPi = pi * 2.


@for_all_methods(profile)
class ParticleFilter:
    def __init__(self, computeWeightsF, physicsUpdateFunction, initializeStatesDiffuseAndWeights, stochasticDiffusionFunction, estimateFunction):
        
        self.initial_measurement_noise = 1.0
        self.decreaseMeasurementNoise = False
        self.computeWeightsF = computeWeightsF
        self.physicsUpdate = physicsUpdateFunction
        self.initializeStatesDiffuseAndWeights = initializeStatesDiffuseAndWeights
        self.stochasticDiffusionFunction = stochasticDiffusionFunction
        self.estimateFunction = estimateFunction
        self.Z = None
        self.Weights = None
        self.i = None
    
    def setMeasurementNoise(self, measurement_noise):
        self.measurement_noise = measurement_noise
        self.sigma_sq_times_2 = (measurement_noise ** 2) * 2.
        self.log_sqrt_twoPi_sigmaSq = np.log(sqrt(twoPi * self.sigma_sq_times_2/2.))
    
    def tryInitialize(self, n):
        if self.Weights is None:
            self.States, self.Diffuse, self.Weights = self.initializeStatesDiffuseAndWeights(n=n, Z = self.Z)
            self.i = 0
            self.estimates = []
    def resample(self, n):
        
        indices = np.arange(len(self.Weights))
        weighted_indices, Weights = stochastic_universal_sampling_logs(indices, self.Weights, n)   
        
        #choose rows of the state array
        self.States = np.take(self.States, weighted_indices, axis = 0)
            
    def particle_filter_step(self, i, n=5, decreaseNoise = None, skipDiffuseStep=False):
        self.tryInitialize(n)

        if i != 0:
            self.resample(n)
        
        if decreaseNoise is not None:
            self.Diffuse[:] *= decreaseNoise
        
        if i != 0:
            self.physicsUpdate(self.States, i)
            
        self.lastEstimate = self.estimateFunction(self.States)
        
        if not skipDiffuseStep:
            self.stochasticDiffusionFunction(self.States, self.Diffuse)
        # compute new weights based on performance of particle
        self.Weights = self.computeWeightsF(self.States, self.Z, i)

    def addObservations(self, z):
        if self.Z is None:
            self.Z = []
        self.Z.extend(z)
    
    def stepAndReturnEstimate(self, decreaseNoise = None, skipDiffuseStep=False, n = None):
        self.tryInitialize(n)
        
        self.particle_filter_step(self.i, n=n, decreaseNoise = decreaseNoise, skipDiffuseStep=skipDiffuseStep)
        self.i += 1
        return self.lastEstimate
        
    
        


