"""
This script is used to simualte emission for high and low damage.
"""
# packages
import os, sys
sys.path.append(os.path.dirname(os.getcwd()) + '/source')

import numpy as np
import matplotlib.pyplot as plt
from simulation import simulate_emission_quadratic, simulate_log_damage
import pickle
import time
##################start of the simulation########################

# model parameters
δ = 0.01
η = 0.032
median = 1.75/1000
h_hat = 0.2
σ_n = 1.2
γ_low = 0.012
γ_high = 0.024
ξ = σ_n/h_hat*δ*η


for γ in [\gamma]
