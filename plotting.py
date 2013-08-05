import matplotlib as mpl
import math
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import initialization
import utils
mpl.rcParams.update({'figure.autolayout': True})
# print mpl.rcParams.keys()
# mpl.rcParams['animation.frame_format'] = 'jpeg'
# print mpl.rcParams['animation.frame_format']

def plot_list(fig, plot_list):
	"""
	Takes a list of lambda forms of plot functions and plots them such that
	no more than four rows are used to assure readability
	"""
	n_plots = len(plot_list)
	for n, p in enumerate(plot_list, start=1):
		if n_plots < 4:
			fig.add_subplot(n_plots, 1, n)
		else:
			fig.add_subplot(math.ceil(n_plots/2.), 2, n)
		p()

# def set_rates(self, position):
# 	"""
# 	Computes the values of all place field Gaussians at <position>

# 	Future Tasks:
# 		- 	Maybe do not normalize because the normalization can be put into the
# 			weights anyway
# 		- 	Make it work for arbitrary dimensions
# 	"""
# 	if self.dimensions == 1:
# 		rates = self.norm*np.exp(-np.power(position - self.centers, 2)*self.twoSigma2)
# 		return rates

def set_current_output_rate(self):
	"""
	Sums exc_weights * exc_rates and substracts inh_weights * inh_rates
	"""
	rate = (
		np.dot(self.exc_syns.weights, self.exc_syns.rates) -
		np.dot(self.inh_syns.weights, self.inh_syns.rates)
	)
	self.output_rate = utils.rectify(rate)

def set_current_input_rates(self):
	"""
	Set the rates of the input neurons by using their place fields
	"""
	self.exc_syns.set_rates(self.x)
	self.inh_syns.set_rates(self.x)

class Plot:
	"""The Plotting Class"""
	def __init__(self, params, rawdata):
		for k, v in params.items():
			setattr(self, k, v)				
		for k, v in rawdata.items():
			setattr(self, k, v)
		self.box_linspace = np.linspace(0, params['boxlength'], 200)
		self.time = np.arange(0, self.simulation_time + self.dt, self.dt)
		self.colors = {'exc': 'g', 'inh': 'r'}
		# self.fig = plt.figure()

	def set_rates(self, position, norm, sigma, twoSigma2, centers):
		"""
		Computes the values of all place field Gaussians at <position>

		Future Tasks:
			- 	Maybe do not normalize because the normalization can be put into the
				weights anyway
			- 	NOTE: This is simply take from the Synapse class, but now you use
				it with an additional argument <syn_type> to make it easier to use here
		"""
		# if self.dimensions == 1:
		rates = norm*np.exp(-np.power(position - centers, 2)*twoSigma2)
		return rates

	def output_rates_vs_positions(self):
		_positions = self.positions[:,0]
		plt.plot(_positions, self.output_rates, linestyle='none', marker='o')

	def output_rate_as_function_of_fields_and_weights(self, time=-1):
		"""docstring"""
		# Get the rates
		n_values = 201
		linspace = np.linspace(0, self.boxlength, n_values)
		rates = {'exc': [], 'inh': []}
		for x in linspace:
			for syn_type in ['exc', 'inh']:
				_sigma = getattr(self, 'sigma_' + syn_type)
				twoSigma2 = 1. / (2 * _sigma**2)
				norm = 1. / (_sigma * np.sqrt(2 * np.pi))
				centers = getattr(self, syn_type + '_centers')	
				rates[syn_type].append(self.set_rates(x, norm, _sigma, twoSigma2, centers))
		output_rates = np.zeros(n_values)
		for n, x in enumerate(linspace):
			output_rates[n] = (np.dot(self.exc_weights[time], rates['exc'][n]) 
							- np.dot(self.inh_weights[time], rates['inh'][n]))
		output_rates = utils.rectify_array(output_rates)
		plt.title('output_rate_as_function_of_fields_and_weights, Time = ' + str(time))
		plt.plot(linspace, output_rates)
	# def output_rate_as_function_of_fields_and_weights(self):
	# 	"""docstring"""
	# 	pass

	def fields_times_weights(self, time=-1, syn_type='exc', normalize_sum=True):
		"""
		Plots the Gaussian Fields multiplied with the corresponding weights

		Arguments:
		- time: default -1 takes weights at the last moment in time
				Warning: if time_step != 1.0 this doesn't work, because
				you take the array at index [time]
		- normalize_sum: If true the sum gets scaled such that it
			is comparable to the height of the weights*gaussians,
			this way it is possible to see the sum and the individual
			weights on the same plot. Otherwise the sum would be way larger.
		"""
		plt.title(syn_type + ' fields x weights')
		x = self.box_linspace
		t = syn_type
		# colors = {'exc': 'g', 'inh': 'r'}	
		summe = 0
		divisor = 1.0
		if normalize_sum:
			# divisor = 0.5 * len(rawdata[t + '_centers'])
			divisor = 0.5 * len(getattr(self, t + '_centers'))			
		for c, s, w in np.nditer([
						getattr(self, t + '_centers'),
						getattr(self, t + '_sigmas'),
						getattr(self, t + '_weights')[time]	]):
			gaussian = scipy.stats.norm(loc=c, scale=s).pdf
			l = plt.plot(x, w * gaussian(x), color=self.colors[syn_type])
			summe += w * gaussian(x)
		plt.plot(x, summe / divisor, color=self.colors[syn_type], linewidth=4)
		return l

	def fields(self, show_sum=False):
		"""
		Plotting of Gaussian Fields and their sum

		Note: The sum gets divided by a something that depends on the 
				number of cells of the specific type, to make it fit into
				the frame (see note in fields_times_weighs)
		"""
		x = self.box_linspace
		# Loop over different synapse types and color tuples
		for t, color in [('exc', 'g'), ('inh', 'r')]:
			summe = 0
			for c, s in np.nditer([getattr(self, t + '_centers'), getattr(self, t + '_sigmas')]):
				gaussian = scipy.stats.norm(loc=c, scale=s).pdf
				plt.plot(x, gaussian(x), color=color)
				summe += gaussian(x)
			if show_sum:
				plt.plot(x, 2*summe/(len(getattr(self, t + '_centers'))), color=color, linewidth=4)
		return

	def weights_vs_centers(self, syn_type='exc', time=-1):
		plt.title(syn_type + ' Fields vs Centers' + ', ' + 'Time = ' + str(time))	
		plt.xlim(0, self.boxlength)
		centers = getattr(self, syn_type + '_centers')
		weights = getattr(self, syn_type + '_weights')[time]
		plt.plot(centers, weights, linestyle='none', marker='o')

	def weight_evolution(self, syn_type='exc'):
		"""
		Plots the time evolution of each synaptic weight
		"""
		plt.title(syn_type + ' weight evolution')
		time = np.arange(0, len(self.exc_weights)) * self.every_nth_step
		for i in np.arange(0, getattr(self, 'n_' + syn_type)):
			# Create array of the i-th weight for all times
			weight = getattr(self, syn_type + '_weights')[:,i]
			plt.plot(time, weight)

	def output_rate_distribution(self, n_last_steps=10000):
		n_bins = 100
		positions = self.positions[:,0][-n_last_steps:,]
		output_rates = self.output_rates[-n_last_steps:,]
		dx = self.boxlength / n_bins
		bin_centers = np.linspace(dx, self.boxlength-dx, num=n_bins)
		mean_output_rates = []
		for i in np.arange(0, n_bins):
			indexing = (positions >= i*dx) & (positions < (i+1)*dx)
			mean_output_rates.append(np.mean(output_rates[indexing]))
		plt.plot(bin_centers, mean_output_rates, marker='o')
		plt.axhline(y=self.target_rate, linewidth=3, linestyle='--', color='black')

	def position_distribution(self):
		x = self.positions[:,0]
		n, bins, patches = plt.hist(x, 50, normed=True, facecolor='green', alpha=0.75)