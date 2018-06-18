import numpy as np
from gridscore import spikes
from gridscore import plotting
from scipy.stats import circmean
import matplotlib.pyplot as plt

##########################################################################
####################### Spike data from grid cells #######################
##########################################################################

# Data of a 10 minute recording.
# Precisely: recording 11016-31010502_T6C3 from Sargolini et al. 2006.
# Load spikepositions for this recording.
spikepositions = np.load('spikepositions.npy')

# Spiketimes of the spikepositions.
# Irrelevant for the final grid score, but needed for plotting the
# time evolution of the grid score (if desired).
# spiketimes = np.load('spiketimes.npy')
# If time data don't matter or don't exist, it can be set to None.
spiketimes = None

# The arena in the Sargolini 2006 experiments is quadratic and
# between -50 cm and 50 cm in both dimensions.
arena_limits = np.array([[-50, 50], [-50, 50]])
# Initiating a Spikes class
spks = spikes.Spikes(positions=spikepositions, arena_limits=arena_limits,
					 times=spiketimes)

# PSI: A complex number for each spike
psi = spks.psi()

# The absolute value of psi gives the local grid score of each spike
psi_abs = np.absolute(psi)

# The global PSI grid score is the mean of the individual scores
global_gridscore = np.mean(psi_abs)
print('PSI gridscore:', global_gridscore)

# The direction of psi gives the local orientation of the grid around each spike
# All elements with psi strictly zero did not pass the comparison with
# other symmetries. These 0 would flawfully indicate 0 orienation.
# We therefore set them to NaN.
psi = spks.psi(compare_to_other_symmetries=False)

# Each angle between two neighboring spikes is increased in psi
# by a factor of 6. To get the local orientation, we need to
# undo this rotation by dividing by 6.
# NB: np.angle returns angles in the range -Pi to +Pi.
# So for n_symmetry = 6 the line below leads to
# angles between -30 and 30 degrees.
psi_angle = np.angle(psi, deg=True) / 6

# The global grid orientation is the circular mean of the local orientations
global_orientation = circmean(psi_angle, high=30, low=-30)
print('Global grid orientation:', global_orientation)

###############################################
################## Plotting ###################
###############################################
plt.figure(figsize=(3, 6))
# The size of the symbols for each spike in pts
dotsize = 7

# Instantiating the Plot class
plot = plotting.Plot(spikepositions=spikepositions, arena_limits=arena_limits)

# Local grid scores
plt.subplot(311)
plot.spikemap(shell_limits_choice='automatic_single', dotsize=dotsize)

# Local orientations
plt.subplot(312)
plot.spikemap(shell_limits_choice='automatic_single',
			  compare_to_other_symmetries=False,
			  color_code='psi_angle', dotsize=dotsize)

# plt.figure(figsize=(2, 2))
# Histogram of distances between all locations with highlighted shell.
plt.subplot(313)
plot.distance_histogram()

plt.savefig('figures/spikepositions.png', dpi=100, bbox_inches='tight')
plt.show()
