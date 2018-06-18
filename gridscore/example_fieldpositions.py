import numpy as np
from gridscore import spikes
from gridscore import plotting
from scipy.stats import circmean
import matplotlib.pyplot as plt
import scipy.io as sio

#########################################################################
####################### Data with field locations #######################
#########################################################################

# Instead of taking the second peak in the histogram (default maximum_index = 1)
# we take the first peak
maximum_index = 0
# We weight each peak in the histogram of distances between all locations
# by the distance.
# This make it trivial to obtain the first peak.
weights = 'inverse_distance'

# Load spikepositions for this recording
# We need an array of shape (N, 2)
# The array in the matlab file is of shape (2, N). We therefore take
# the transpose.
spikepositions = sio.loadmat('positions.mat')['positions'].T

# The arena ranges from 0 to 10 along both dimensions.
arena_limits = np.array([[0, 10], [0, 10]])

# Initiating a Spikes class
spks = spikes.Spikes(positions=spikepositions, arena_limits=arena_limits)

# PSI: A comple number for each spike
psi = spks.psi(weights=weights, maximum_index=maximum_index)

# The absolute value of psi give the grid score of each spike
psi_abs = np.absolute(psi)

# The global PSI grid score is the mean of the individual scores
global_gridscore = np.mean(psi_abs)
print('PSI gridscore:', global_gridscore)

# The direction of psi give the local orientation of the grid around each spike
# See explanation above.
psi_angle = np.angle(psi, deg=True) / 6

# The global grid orientation is the circular mean of the local orientations
# See explanation above.
global_orientation = circmean(psi_angle[np.isfinite(psi_angle)],
							  high=30, low=-30)
print('Global grid orientation:', global_orientation)

###############################################
################## Plotting ###################
###############################################
plt.figure(figsize=(4, 6))
dotsize = 7

# Instantiating the Plot class
plot = plotting.Plot(spikepositions=spikepositions, arena_limits=arena_limits)

# Local grid scores
plt.subplot(311)
plot.spikemap(shell_limits_choice='automatic_single', weights=weights,
			  dotsize=dotsize, compare_to_other_symmetries=True,
			  maximum_index=maximum_index)

# Local grid orientations
plt.subplot(312)
plot.spikemap(shell_limits_choice='automatic_single',
			  compare_to_other_symmetries=False,
			  color_code='psi_angle', weights=weights,
			  dotsize=dotsize, maximum_index=maximum_index)

# Histogram of distances between all locations with highlighted shell.
plt.subplot(313)
plot.distance_histogram(weights=weights, maximum_index=maximum_index)

plt.savefig('figures/fieldpositions.png', dpi=100, bbox_inches='tight')
plt.show()
