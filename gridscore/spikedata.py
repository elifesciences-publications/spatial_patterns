import os

import numpy as np
import scipy.io as sio
import scipy.stats as stats

import general_utils.arrays
import general_utils.snep_plotting as snep_plotting
from learning_grids.utils import check_conditions
from .artificial_ratemaps import ArtificialRatemaps
from .dictionaries import filenames_sargolini2006
from .dictionaries import filenames_hafting2005
import pandas as pd

MAINDIR = '/Users/simonweber/doktor/Data/'

PATHS_DICTIONARY = {
	'hafting2005': 'Hafting_2005/hafting_data/',
	'sargolini2006': 'Sargolini_2006/8F6BE356-3277-475C-87B1'
					 '-C7A977632DA7_1/all_data/',
	'escobar2016': '2016_Escobar/circular_arena'
}

class SpikeData:
	"""
	Class to handle raw experimental data

	This class is mainly used to deal with the data structure of
	the Moser lab.
	The structure varies slightly between and within publications.
	The main goal is to obtain the following arrays:
	ratpositions : ndarray
		The positions of the rat over time; shape = ()
	positiontimes : ndarray
		The time at each of the rat positions; shape = ()
	spiketimes : ndarray
		The time of each spike; shape = ()


	Parameters
	----------
	publication : str
		e.g.,
		hafting2005
	filename : str
		For hafting_2005:
			figure2d_trial1

	identifier : str
		e.g.,
		'rat11015_t1c1'

	Notes
	-----
	publication, filename and identifier only work in certain combinations
	Moreover, the nomination is not very systematic.
	Here are some possible combinations with a qualitative description:

	publication:
		hafting_2005
			filename:
				figure2d_trial1
					identifier:
						rat11015_t1c1
							Good grid cell
						rat11015_t2c1
							Decent grid cell
						rat11015_t2c2

						-- complete --

				figure2c_trial1
					identifier:
						rat11015_t5c1_timeStamps
							Noisy grid cell
						rat11015_t5c2_timeStamps
							Bad grid cell
						rat11015_t5c3_timeStamps
						rat11015_t6c1_timeStamps
						rat11015_t7c1_timeStamps
						rat11015_t7c2_timeStamps
						rat11015_t7c3_timeStamps
						... more ...

	"""

	def __init__(self,
				 maindir=MAINDIR,
				 publication='hafting2005',
				 filename='figure2d_trial1',
				 identifier='rat11015_t1c1'):
		self.identifier = identifier
		self.publication = publication
		self.filename = filename
		paths = PATHS_DICTIONARY
		arena_limits = {
			'hafting2005': np.array([[-90, 90], [-90, 90]]),
			'sargolini2006': np.array([[-50, 50], [-50, 50]]),
			'escobar2016': np.array([[0, 800], [0, 800]]),
		}
		if publication == 'hafting2005':
			filenames = filenames_hafting2005
			self.path = os.path.join(maindir, paths[publication],
									 filenames[filename])
			self.data = sio.loadmat(self.path)
			self.ratpositions, self.positiontimes, self.spiketimes, \
			self.ratpositions2, self.headdirections = \
				self.get_ratpositions_positiontimes_spiketimes_hafting()
		elif publication == 'sargolini2006':
			filenames = filenames_sargolini2006
			path_spiketimes = os.path.join(maindir, paths[publication],
										   filenames[filename][0])
			path_positions = os.path.join(maindir, paths[publication],
										  filenames[filename][1])
			self.data = np.array([sio.loadmat(path_spiketimes),
								  sio.loadmat(path_positions)])
			self.ratpositions, self.positiontimes, self.spiketimes, \
			self.ratpositions2, self.headdirections = \
				self.get_ratpos_post_spiket_ratpos2_hds_sargolini()
		elif publication == 'escobar2016':
			# filename = 'jp693-11062015-0108'
			topdir = filename.split('-', 1)[0]
			self.path = os.path.join(maindir, paths[publication], topdir,
								   filename)
			self.ratpositions = self.get_ratpositions_escobar()
			self.sampling_rate = self.get_sampling_rate_escobar()
			time_between_positions = (
					self.get_samples_between_position_samples_escobar() /
					self.sampling_rate
			)
			self.positiontimes = self.get_positiontimes_escobar(
				time_between_positions)
			self.spiketimes = self.get_spiketimes_escobar()

		self.arena_limits = arena_limits[publication]

	def get_ratpositions_escobar(self):
		"""Ratpositions in circular arena are obtained from .whl"""
		fname = self.get_fname_from_extension('whl')
		ret = pd.read_csv(fname, delimiter=" ", header=None).as_matrix()
		return ret

	def get_sampling_rate_escobar(self):
		"""Number of electrophisiological samples per second"""
		fname = self.get_fname_from_extension('sampling_rate_dat')
		return pd.read_csv(fname, delimiter=" ", header=None).values[0, 0]

	def get_samples_between_position_samples_escobar(self):
		"""Number of electrophisiological samples between 2 positions samples"""
		fname = self.get_fname_from_extension('res_samples_per_whl_sample')
		return pd.read_csv(fname, delimiter=" ", header=None).values[0, 0]

	def get_positiontimes_escobar(self, time_between_positions):
		"""Position recordings start at time 0 and are uniformly incremented"""
		return np.arange(len(self.ratpositions)) * time_between_positions

	def get_spiketimes_escobar(self):
		"""For the spiketimes a cluster number is seleceted from .clu"""
		fname = self.get_fname_from_extension('clu')
		# From the .clu file you skip the first row, because it contains the
		# number of cluster (including the noise cluster 1).
		clusters = pd.read_csv(fname, delimiter=" ", header=None,
						   skiprows=1).as_matrix()
		fname = self.get_fname_from_extension('res')
		times = pd.read_csv(fname, delimiter=" ", header=None).as_matrix()
		return times[clusters == self.identifier] / self.sampling_rate

	def get_fname_from_extension(self, extension):
		"""Returns the path to a file with a given extension"""
		file = self.filename + '.' + extension
		return os.path.join(self.path, file)

	def get_ratpos_post_spiket_ratpos2_hds_sargolini(self):
		"""
		NB: Determining ratpositions 1 and 2 needs to be intertwined, 
		to handle erraneous data correctly.
		"""
		ratpositions = np.dstack(
			(self.data[1]['posx'][:, 0],
			 self.data[1]['posy'][:, 0])
		)[0, :]
		try:
			# If the rat in the experiments had a second LED attached to
			# its body, a second positions array exists.
			ratpositions2 = np.dstack(
				(self.data[1]['posx2'][:, 0],
				 self.data[1]['posy2'][:, 0])
			)[0, :]
			led2_exists = True
		except IndexError:
			# If no second positions array exist, the code above leads to
			# an index error. To simplify future processing, we keep a
			# second positions array, setting all positions to 0, 0.
			ratpositions2 = np.zeros_like(ratpositions)
			led2_exists = False
		positiontimes = self.data[1]['post'][:, 0]
		spiketimes = self.data[0]['cellTS'][:, 0]
		# Remove rows with NaN ratpositions from positions and
		# positiontimes. This occurs in Sargolini data.
		# We check if the second LED (and thus a second positions array)
		# exist and ensure that the two position arrays have the same shape.
		if led2_exists and (ratpositions.shape == ratpositions2.shape):
			not_nan1 = general_utils.arrays.not_nan_rows_bool(ratpositions)
			not_nan2 = general_utils.arrays.not_nan_rows_bool(ratpositions2)
			not_nan = np.logical_and(not_nan1, not_nan2)
			ratpositions = ratpositions[not_nan]
			ratpositions2 = ratpositions2[not_nan]
			dx = ratpositions[:, 0] - ratpositions2[:, 0]
			dy = ratpositions[:, 1] - ratpositions2[:, 1]
			headdirections = np.arctan2(dy, dx)
		else:
			not_nan = general_utils.arrays.not_nan_rows_bool(ratpositions)
			ratpositions = ratpositions[not_nan]
			ratpositions2 = ratpositions2[not_nan]
			headdirections = np.zeros_like(ratpositions[:, 0])

		positiontimes = positiontimes[not_nan]

		return ratpositions, positiontimes, spiketimes,\
			   ratpositions2, headdirections

	def get_ratpositions_positiontimes_spiketimes_hafting(self):
		ratpositions = np.dstack(
			(self.data['pos_x'][:, 0],
			 self.data['pos_y'][:, 0])
		)[0, :]
		positiontimes = self.data['pos_timeStamps'][:, 0]
		spiketimes = self.data[self.identifier][:, 0]
		# There is no second LED, so no head direction and no second position
		# are recorded. We set them to 0.
		ratpositions2 = np.zeros_like(ratpositions)
		headdirections = np.zeros_like(ratpositions[:, 0])
		return ratpositions, positiontimes, spiketimes,\
			   ratpositions2, headdirections

	def get_spikepositions(self):
		spike_idx_int = self.get_spike_idx_array()
		return self.ratpositions[spike_idx_int]

	def get_spikeheaddirections(self):
		spike_idx_int = self.get_spike_idx_array()
		try:
			spikehd = self.headdirections[spike_idx_int]
		except IndexError:
			spikehd = np.array([])
		return spikehd

	def get_spike_idx_array(self):
		"""
		Returns index array into all recorded values at each spiketime.
		
		NB: `positiontimes` is used as the array that represents all recording 
		times. If a different array, like `ratpositions` would have a 
		differents size than `positiontimes` (for example because some 
		recordings were missing and instead of replacing them with NaN they 
		just aren't there) this method would lead to unpredictable behavior. 
		At least for the Sargolini data, this does not occur.
		"""
		spike_idx_int = []
		for st in self.spiketimes:
			spike_idx_int.append(
				general_utils.arrays.find_nearest(
					self.positiontimes, st, ret='index')
			)
		spike_idx_int = np.asarray(spike_idx_int)
		return spike_idx_int


class SpikeDataFromSimulations(snep_plotting.Snep, SpikeData):
	"""
	Similar to SpikeData, but obtaines the data from simulations

	Parameters
	----------
	maindir : str
		Path to the folder with simulation experiments
	datedir : str
		Name of simulation folder. NB: This works analogously to
		`publication` in SpikeData and in the experiments the datedir
		is stored in the publication parameter.
	identifier : str
		This specifies one paramspace point of the simulation.
		The selection happens in get_psp
	dt : float
		Time step in seconds, between two position measurements.
		For all simulations where we use behavioral rat trajectories, dt=0.02
	firstframe : int, optional
		Specifies which frame, i.e., timestep, should be the first to be
		taken.
	lastframe : int, optional
		Analogous to firstframe.
	Returns
	-------
	"""

	def __init__(self,
				 maindir='/Users/simonweber/experiments'
						 '/experiment_using_snep/',
				 datedir_acronym='3hrs_simulations_1',
				 identifier='seed_centers_16', rate_factor=20, dt=0.02,
				 firstframe=0, lastframe='none'):
		datedir_dict = {
			'3hrs_simulations_1': '2016-12-08-17h39m18s_180_'
								  'minutes_trajectories_1_fps_examples',
			'3hrs_simulations_2': '2016-12-08-14h13m01s_180_'
								  'minutes_trajectories_1_fps_examples',
			'wall_experiment': '2017-11-09-18h37m32s_wernle_seed_55_'
								  'with_trajectory',
		}
		self.params_and_rawdata_is_given = False
		self.identifier = identifier
		self.maindir = maindir
		self.datedir = datedir_dict[datedir_acronym]
		self.rate_factor = rate_factor
		self.dt = dt
		self.firstframe = firstframe
		self.lastframe = lastframe
		self.path, self.tables, self.psps = snep_plotting.get_path_tables_psps(
			self.datedir, project_name='learning_grids'
		)
		psp = self.get_psp()
		self.set_params_rawdata_computed(psp)
		if self.lastframe == 'none':
			self.lastframe = self.rawdata['positions'].shape[0]
		r = self.params['sim']['radius']
		self.arena_limits = np.array([[-r, r], [-r, r]])
		self.ratpositions, self.positiontimes, self.spiketimes = \
			self.get_ratpositions_positiontimes_spiketimes_from_simulations()

	def get_ratpositions_positiontimes_spiketimes_from_simulations(self):
		"""
		Get the required data from the simulation rawdata

		This ensures that the data has the same format, as when obtained
		from experimental data.
		See equally named function in SpikeData
		"""
		x_positions = self.rawdata['positions'][self.firstframe:self.lastframe,
					  0]
		y_positions = self.rawdata['positions'][self.firstframe:self.lastframe,
					  1]
		ratpositions = np.dstack((x_positions, y_positions))[0, :]
		# simtime_in_seconds = self.params['sim']['simulation_time'] * self.dt
		# -1, because the inital positions are recorded
		simtime_in_seconds = (x_positions.size - 1) * self.dt
		positiontimes = np.linspace(0, simtime_in_seconds, x_positions.size)
		t_start = self.firstframe * self.dt
		t_end = self.lastframe * self.dt
		spiketimes_all = self.computed['spiketimes'][str(self.rate_factor)]
		time_condition = np.logical_and(t_end >= spiketimes_all,
										spiketimes_all >= t_start)
		# spiketimes_in_timewindow = spiketimes_all[
		# 								t_end >= spiketimes_all >= t_start]
		spiketimes_in_timewindow = spiketimes_all[time_condition]
		spiketimes = spiketimes_in_timewindow - t_start
		return ratpositions, positiontimes, spiketimes

	def get_psp(self):
		"""
		Returns paramspace points, specified by `datedir` and `identifier`

		The dictionary psp_dict uses the tuple (datedir, identifier) as key
		and has a single psp as value. The single psp needs to be set manually
		here and the naming of the identifier should make sense, accordingly.

		NB: This is not very elegant, but allows you to have full control
		over what simulation (and experimental) spike data to combine in
		a single gridscore experiment.

		Returns
		-------
		psp
		"""
		if self.identifier == 'seed_centers_16':
			conditions = [
				(('sim', 'seed_centers'), 'eq', 16)
			]
		elif self.identifier == 'seed_centers_24':
			conditions = [
				(('sim', 'seed_centers'), 'eq', 24)
			]
		elif self.identifier == 'seed_centers_22':
			conditions = [
				(('sim', 'seed_centers'), 'eq', 22)
			]
		elif self.identifier == 'seed_centers_55':
			conditions = [
				(('sim', 'seed_centers'), 'eq', 55)
			]
		elif self.identifier == 'seed_centers_4':
			conditions = [
				(('sim', 'seed_centers'), 'eq', 4)
			]
		elif self.identifier == 'seed_centers_5':
			conditions = [
				(('sim', 'seed_centers'), 'eq', 5)
			]
		elif self.identifier == 'seed_centers_6':
			conditions = [
				(('sim', 'seed_centers'), 'eq', 6)
			]
		elif self.identifier == 'seed_centers_7':
			conditions = [
				(('sim', 'seed_centers'), 'eq', 7)
			]
		elif self.identifier == 'seed_centers_8':
			conditions = [
				(('sim', 'seed_centers'), 'eq', 8)
			]
		elif self.identifier == 'seed_centers_9':
			conditions = [
				(('sim', 'seed_centers'), 'eq', 9)
			]
		elif self.identifier == 'seed_centers_140':
			conditions = [
				(('sim', 'seed_centers'), 'eq', 140)
			]
		psp = [p for p in self.psps if check_conditions(p, *conditions)][0]
		return psp

	# NB: The below does not work, if you use more than one experiment
	# file
	# psp_dict = {
	# 	('2016-12-08-17h39m18s_180_minutes_trajectories_1_fps_examples',
	# 	 'seed_centers_16'): [
	# 		p for p in self.psps
	# 		 if p[('sim', 'seed_centers')].quantity == 16][0],
	#
	# 	('2016-12-08-17h39m18s_180_minutes_trajectories_1_fps_examples',
	# 	 'seed_centers_24'): [
	# 		p for p in self.psps
	# 		if p[('sim', 'seed_centers')].quantity == 24][0],
	#
	# 	('2016-12-08-14h13m01s_180_minutes_trajectories_1_fps_examples',
	# 	 'seed_centers_4'): [
	# 		p for p in self.psps
	# 		if p[('sim', 'seed_centers')].quantity == 4][0],
	#
	# 	('2016-12-08-14h13m01s_180_minutes_trajectories_1_fps_examples',
	# 	 'seed_centers_5'): [
	# 		p for p in self.psps
	# 		if p[('sim', 'seed_centers')].quantity == 5][0],
	#
	# 	('2016-12-08-14h13m01s_180_minutes_trajectories_1_fps_examples',
	# 	 'seed_centers_6'): [
	# 		p for p in self.psps
	# 		if p[('sim', 'seed_centers')].quantity == 6][0],
	#
	# 	('2016-12-08-14h13m01s_180_minutes_trajectories_1_fps_examples',
	# 	 'seed_centers_7'): [
	# 		p for p in self.psps
	# 		if p[('sim', 'seed_centers')].quantity == 7][0],
	#
	# 	('2016-12-08-14h13m01s_180_minutes_trajectories_1_fps_examples',
	# 	 'seed_centers_8'): [
	# 		p for p in self.psps
	# 		if p[('sim', 'seed_centers')].quantity == 8][0],
	#
	# 	('2016-12-08-14h13m01s_180_minutes_trajectories_1_fps_examples',
	# 	 'seed_centers_9'): [
	# 		p for p in self.psps
	# 		if p[('sim', 'seed_centers')].quantity == 9][0],
	#
	# 	('2016-12-08-14h13m01s_180_minutes_trajectories_1_fps_examples',
	# 	 'seed_centers_140'): [
	# 		p for p in self.psps
	# 		if p[('sim', 'seed_centers')].quantity == 140][0],
	#
	# }
	# return psp_dict[(self.datedir, self.identifier)]


class SpikeDataExisting():
	def __init__(self,
				 filename='jaeson',
				 identifier='full_hex'):
		files = dict(
			jaeson=sio.loadmat('/Users/simonweber/doktor/Data/'
							   'jaeson_retinal_cells/example_mosaic_Jang.mat'),
			playground=np.load('/Users/simonweber/programming/'
							   'workspace/gridscore/spikepositions.npy'),

		)
		arena_limits_dict = dict(
			jaeson=np.array([[0, 2000], [0, 2000]]),
			playground=np.array([[-50, 50], [-50, 50]]),
		)
		self.arena_limits = arena_limits_dict[filename]
		# For structured arrays, further indexing can be done using `identifier`
		if identifier == 'none':
			self.spikepositions = files[filename]
		else:
			self.spikepositions = files[filename][identifier]
		self.n_spikes = self.spikepositions.shape[0]
		self.ratpositions, self.positiontimes, self.spiketimes = \
			self.get_ratpositions_positiontimes_spiketimes()

	def get_ratpositions_positiontimes_spiketimes(self):
		"""
		Create useless data here to comply with the format

		See equally named function in SpikeData
		"""
		ratpositions = np.array([[np.nan, np.nan]])
		positiontimes = ratpositions
		spiketimes = np.arange(self.n_spikes)
		# np.random.shuffle(spiketimes)
		return ratpositions, positiontimes, spiketimes

	def get_spikepositions(self):
		return self.spikepositions


class SpikeDataArtificial(ArtificialRatemaps):
	def __init__(self,
				 distribution='whitenoise',
				 gridspacing=0.15,
				 fieldsize=0.02,
				 orientation=0.0,
				 fieldlocation_noise=0.0,
				 n_spikes=2000,
				 field_factor_noise=0.0,
				 locality_of_rotation_angle='none',
				 local_noise='none',
				 shuffle_positions=False):
		if locality_of_rotation_angle == 'none':
			locality_of_rotation_angle = None
		if local_noise == 'none':
			local_noise = None
		ArtificialRatemaps.__init__(self,
									distribution=distribution,
									gridspacing=gridspacing,
									fieldsize=fieldsize,
									fieldlocation_noise=fieldlocation_noise,
									orientation=orientation,
									field_factor_noise=field_factor_noise,
									locality_of_rotation_angle=locality_of_rotation_angle,
									local_noise=local_noise)
		self.shuffle_positions = shuffle_positions
		# self.distribution = distribution
		self.n_spikes = n_spikes
		self.arena_limits = np.array([[0, 1], [0, 1]])
		# artificial_rms = ArtificalRatemaps(distribution=distribution)
		# self.rm = artificial_rms.rm
		self.ratpositions, self.positiontimes, self.spiketimes = \
			self.get_ratpositions_positiontimes_spiketimes_artifically()

	def get_spikepositions(self, only_unique_positions=False):
		sfr = SpikesFromRatemap(ratemap=self.rm_list, arena_limits=self.arena_limits)
		spikepositions = sfr.get_spikepositions(self.n_spikes, noise=0)

		if only_unique_positions:
			spikepositions = spikepositions[
				np.unique(spikepositions[:, 0], return_index=True, axis=0)[1]]

		if self.shuffle_positions:
			np.random.shuffle(spikepositions)
		return spikepositions

	def get_ratpositions_positiontimes_spiketimes_artifically(self):
		"""
		Create useless data here to comply with the format

		It does not make sense to speak of ratpositions and positiontimes,
		if the spikes are drawn from an artifical ratemap
		It does not make sense to speak of spiketimes either, but we
		return a shuffled array of the same length as spikepositions

		See equally named function in SpikeData
		"""
		ratpositions = np.array([[np.nan, np.nan]])
		positiontimes = ratpositions
		spiketimes = np.arange(self.n_spikes)
		# np.random.shuffle(spiketimes)
		return ratpositions, positiontimes, spiketimes


class SpikesFromRatemap:
	"""
	Class to get spike locations from rate map

	NOTE: it currently only works for quadratic or circular arenas
	that are defined between 0 and limit
	For rectangular arenas you would need to consider arena limits
	in the index2location function more carefully.

	Parameters
	----------
	ratemap : list of (N, N) ndarrays of length M
	arena_limits : (2, 2) ndarray
		x limits and y limits

	Returns
	-------
	"""

	def __init__(self, ratemap, arena_limits):
		self.rm = ratemap
		self.arena_limits = arena_limits
		self.limit = arena_limits[0, 1]

	def get_index_distributions(self):
		"""
		Returns distribution of indeces, taking the rate map as distribution

		Returns
		-------
		idx_dist :
		"""
		idx_dist_list = []
		for rm in self.rm:
			# Get a value for each rate map element
			xk = np.arange(rm.size)
			# Obtain and normalize a probablity for each xk
			# using the firing rates in the rate map
			pk = np.ravel(rm)
			pk /= np.sum(pk)
			idx_dist = stats.rv_discrete(
				a=xk[0], b=xk[-1], name='idx_dist',
				values=(xk, pk)
			)
			idx_dist_list.append(idx_dist)
		return idx_dist_list

	def draw_indices(self, n):
		"""
		Draw indices from the distribution given by the rate map

		Parameters
		----------
		n : int
			Number of locations to be drawn

		Returns
		-------
		idx : (n,) ndarray
			Array of indices
		"""
		idx_dist_list = self.get_index_distributions()
		n_per_ratemap = int(n / len(self.rm))
		idx = np.array([], dtype=np.int64)
		for idx_dist in idx_dist_list:
			idx2 = idx_dist.rvs(size=n_per_ratemap)
			idx = np.concatenate((idx, idx2))
		return idx

	def indices2location(self, idx):
		"""

		Parameters
		----------
		idx : (N^2) ndarray
			Array of indices

		Returns
		-------
		positions : (N^2, 2) ndarray
		"""
		rows, columns = np.unravel_index(idx, self.rm[0].shape)

		factor = self.limit / float(self.rm[0].shape[0])
		# TODO: maybe switch columns and rows
		# Shouldn't matter for quadratic enclosures
		return np.dstack((columns * factor, rows * factor))[0, :]

	def get_spikepositions(self, n, noise=None):
		"""
		Returns spikes positions

		Parameters
		----------
		n : int
			Number of positions
		noise : float
			Standard deviation of Gaussian noise

		Returns
		-------
		locations : (n, 2) ndarray
		"""
		idx = self.draw_indices(n)
		locations = self.indices2location(idx)
		if noise:
			radial_noise = np.random.randn(n) * noise
			angular_noise = np.random.randn(n) * 2 * np.pi
			noise_shifts_x = radial_noise * np.cos(angular_noise)
			noise_shifts_y = radial_noise * np.sin(angular_noise)
			locations[:, 0] += noise_shifts_x
			locations[:, 1] += noise_shifts_y
		return locations


def print_data_dictionary_sargolini():
	"""
	Used to get a dictionary of filenames, as required in `spikedata.py`
	
	Since this needs to be done only once per entire dataset of a group
	it is good enough to do it manually.
	
	Prints the dictionary. Copy and paste it to filenames in `SpikeData` class.
	"""
	maindir = '/Users/simonweber/doktor/Data/'
	datadir = 'Sargolini_2006/8F6BE356-3277-475C-87B1-C7A977632DA7_1/all_data/'
	fulldir = os.path.join(maindir, datadir)

	# List of all filenames that contain positions
	position_files = []
	# List of all filenames
	all_files = []
	for f in os.listdir(fulldir):
		if is_position_file(f):
			position_files.append(f)
		all_files.append(f)

	# The dictionary that will be used as the filenames dictionary in SpikeData
	d = {}
	# List of string that uniquely identify each recorded cell
	sesssion_tetrode_cell_list = []
	for pf in position_files:
		# Session ID that corresponds to these positions
		session_id = get_session_id(pf)
		# Now we want to find all the recorded cells that were recorded with
		# this trajectory.
		for f in all_files:
			# Check if file is part of this session
			if is_matching_tetrode_cell_file(f, session_id):
				# File that contains spike data
				spikes_file = f
				# Unique identifier of this recording
				session_tetrode_cell = get_session_tetrode_cell_lower_case(f)
				sesssion_tetrode_cell_list.append(session_tetrode_cell)
				d[session_tetrode_cell] = [spikes_file, pf]

	print(d)
	print('*******************************************************')
	print('*******************************************************')
	print('*******************************************************')
	print(sorted(sesssion_tetrode_cell_list))

def is_matching_tetrode_cell_file(filename, session_id):
	"""Check if the filename is of the same session"""
	s = session_id + '_T'
	b = (s in filename) and (filename.endswith('.mat'))
	return b

def get_session_tetrode_cell_lower_case(filename):
	"""Get session id with tetrode and cell number as lower case string"""
	return filename.partition('.mat')[0].lower()

def get_session_id(filename):
	"""Get the session id string from the filename"""
	return filename.partition('_POS')[0]

def is_position_file(filename):
	"""Check if filename is of a file with position data"""
	return 'POS' in filename and filename.endswith('.mat')
