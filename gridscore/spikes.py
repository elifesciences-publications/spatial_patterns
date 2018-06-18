import numpy as np
import scipy.ndimage as ndimage
import scipy.spatial.distance as dist
from . import arrays
from gridscore import correlogram

def get_psi_n(reference_position, surround_positions, n=6,
			  std_threshold=None, compare_to_other_symmetries=False):
	"""
	Determines the psi n correlation factor
	
	Parameters
	----------
	reference_position : ndarray
		Reference position [x0, y0]
	surround_positions : ndarray
		Positions of surrounding particles
		[[x1, y1], [x2, y2], ...]
	n : int
		Psi value for n-fold symmetry is returned.
	normalization : str
		'surround_positions' : Each value is divided by the number of
		surround positions.
		'none' : The psi values are not normalized here.
	Returns
	-------
	psi : complex float
	"""
	angles = np.arctan2(surround_positions[:, 1] - reference_position[1],
						surround_positions[:, 0] - reference_position[0])
	if std_threshold:
		std = np.std(angles)
		if std < std_threshold:
			return 0
	if surround_positions.size == 0:
		return 0
	if not compare_to_other_symmetries:
		psi = np.sum(np.exp(1j * n * angles)) / surround_positions.shape[0]
	elif compare_to_other_symmetries == 4:
		psi_abs = (
				np.absolute(np.sum(np.exp(1j * 6 * angles)) /
				            surround_positions.shape[0])
				- np.absolute(np.sum(np.exp(1j * 4 * angles)) /
				              surround_positions.shape[0])
		)
		# Make it complex again, but with orientation 0
		psi = psi_abs * np.exp(1j*0)
	else:
		psis = []
		symmetries = np.array([2, 3, 4, 5, 6, 7])
		for n_sym in symmetries:
			psis.append(np.sum(np.exp(1j * n_sym * angles))
						/ surround_positions.shape[0])
		psis = np.asarray(psis)
		psis_abs = np.absolute(psis)
		idx_with_symmetry_n = np.argwhere(symmetries == n)[0, 0]
		idx_with_maximum_psi_abs = np.argmax(psis_abs)
		if idx_with_maximum_psi_abs == idx_with_symmetry_n:
			psi = psis[idx_with_symmetry_n]
		else:
			psi = 0
	return psi


class Spikes:
	"""
	Class that extracts information from spike positions

	Parameters
	----------
	positions : ndarray
		Spike positions
	times : ndarray
		Spike times. Optional, because not essential for most computations.

	Returns
	-------
	"""

	def __init__(self, positions, arena_limits, times=None):
		self.arena_limits = arena_limits
		self.radius = arena_limits[0, 1]
		# x_shape, y_shape = positions.shape
		# noise = 2 * noise * np.random.random_sample(positions.shape) - noise
		self.pos = positions
		self.n_symmetry = 6
		self.times = times
		self.n_spikes = self.pos.shape[0]

	def get_spiketimes_spikenumbers(self, every_nth=1, include_zero=False):
		n_spikes = len(self.times)
		spiketimes = self.times[every_nth-1::every_nth]
		spikenumbers = np.arange(every_nth, n_spikes+1, every_nth)
		if include_zero:
			spiketimes = np.insert(spiketimes, 0, 0.0)
			spikenumbers = np.insert(spikenumbers, 0, 0)
		return spiketimes, spikenumbers

	def psi(self, n_symmetry=6,
			shell_limits_choice='automatic_single',
			gridscore_norm=None, std_threshold=None,
			compare_to_other_symmetries=True,
			given_shell_limits=None,
			cut_off_position=0,
			weights=None,
			maximum_index=1):
		"""
		Returns the grid score

		Parameters
		----------
		method : str
			'automatic_single', grid score from psi_6, with single shell
			that is determined automatically
			'automatic_double', same but with double shell
			'sargolini', 'Weber' grid score based on correlogram
		Returns
		-------
		grid score : float
		"""
		if given_shell_limits:
			shell_limits = given_shell_limits
		else:
			shell_limits = self.get_shell_limits(
				shell_limits_choice=shell_limits_choice,
				cut_off_position=cut_off_position, weights=weights,
				maximum_index=maximum_index)
		psi = self.psi_n_all(n_symmetry=n_symmetry,
					 shell_limits=shell_limits,
					 normalization=gridscore_norm,
					 std_threshold=std_threshold,
					 compare_to_other_symmetries=compare_to_other_symmetries)
		return psi



	def get_gridscore(self, method='automatic_single', sigma=5,
					  gridscore_norm=None, std_threshold=None,
					  return_std=False,
					  drop_recalc=0,
					  given_shell_limits=None,
					  n_symmetry=6,
					  compare_to_other_symmetries=False):
		"""
		Returns the grid score

		Parameters
		----------
		method : str
			'automatic_single', grid score from psi_6, with single shell
			that is determined automatically
			'automatic_double', same but with double shell
			'sargolini', 'Weber' grid score based on correlogram
		Returns
		-------
		grid score : float
		"""
		if method == 'automatic_single' or method == 'automatic_double'\
				or method == 'given_shell_limits':
			# Create index array with all indeces
			idx = np.arange(self.pos.shape[0])
			for i in np.arange(drop_recalc + 1):
				# Get the spikes positions at the indices
				self.pos = self.pos[idx]
				# if n_try_shell_limits is None:
				if method == 'given_shell_limits':
					shell_limits = given_shell_limits
				else:
					shell_limits = self.get_shell_limits(
						shell_limits_choice=method)
				psi = self.psi_n_all(n_symmetry=n_symmetry,
									 shell_limits=shell_limits,
									 normalization=gridscore_norm,
									 std_threshold=std_threshold,
					 compare_to_other_symmetries=compare_to_other_symmetries)
				psi_abs = np.absolute(psi)
				sort_idx = np.argsort(psi_abs)
				# Keep only indices for spikes with good indices
				good_idx = sort_idx[int(np.floor(len(sort_idx) / 2)):]
				idx = idx[good_idx]
			gridscore = np.mean(psi_abs)
			gridscore_std = np.std(psi_abs)

		elif method == 'sargolini' or method == 'Weber' or method == 'langston':
			import learning_grids.observables as lg_observables
			rm = self.get_ratemap(sigma=sigma)
			corr_spacing, a = correlogram.get_correlation_2d(
				rm, rm, mode='same')
			radius = self.arena_limits[0, 1]
			gridness = correlogram.Gridness(
				a, radius=radius, method=method,
				neighborhood_size=10, threshold_difference=0.2,
				n_contiguous=100)
			try:
				gridscore = gridness.get_grid_score()
			except ValueError as e:
				print('Error in computation of gridscore from correlogram')
				print(e)
				print('returned NaN instead')
				gridscore = np.nan
			gridscore_std = np.nan
		if return_std:
			return gridscore, gridscore_std
		else:
			return gridscore

	def _neighbor_indeces(self, idx):
		idx_all = np.arange(self.n_spikes)
		n_shells = self.get_number_of_shells(self.shell_limits)
		if n_shells == 0:
			# Remove itselfs
			neighbor_idx = np.delete(idx_all, idx, axis=0)
		elif n_shells == 1:
			ref_pos = self.pos[idx]
			dist_to_ref_pos = dist.cdist(np.atleast_2d(ref_pos),
										 self.pos, 'euclidean')[0, :]
			# Removing itsels is not necessary, because the inner shell
			# takes care of it.
			condition_inner = dist_to_ref_pos >= self.shell_limits[0]
			condition_outer = dist_to_ref_pos <= self.shell_limits[1]
			conditions = np.logical_and(condition_inner, condition_outer)
			neighbor_idx = np.argwhere(conditions)[:, 0]
		elif n_shells == 2:
			ref_pos = self.pos[idx]
			dist_to_ref_pos = dist.cdist(np.atleast_2d(ref_pos),
										 self.pos, 'euclidean')[0, :]
			condition_inner_1 = dist_to_ref_pos >= self.shell_limits[0, 0]
			condition_outer_1 = dist_to_ref_pos <= self.shell_limits[0, 1]
			condition_inner_2 = dist_to_ref_pos >= self.shell_limits[1, 0]
			condition_outer_2 = dist_to_ref_pos <= self.shell_limits[1, 1]
			conditions = np.logical_or(
				np.logical_and(condition_inner_1, condition_outer_1),
				np.logical_and(condition_inner_2, condition_outer_2)
			)
			neighbor_idx = np.argwhere(conditions)[:, 0]
		return neighbor_idx

	def get_number_of_shells(self, shell_limits):
		if shell_limits is None:
			n_shells = 0
		else:
			n_shells = len(shell_limits.shape)
		return n_shells

	def get_cut_off_position(self, arena_fraction=0.15):
		cut_off_position = (
			self.arena_limits[0, 1] - self.arena_limits[1, 0]) * arena_fraction
		return cut_off_position

	def get_shell_limits(self, shell_limits_choice=None,
						 neighborhood_size_as_fraction=0.1, cut_off_position=0,
						 threshold_difference_as_fraction=0.05, bins=100,
						 weights=None, maximum_index=1):
		"""
		Determines meaningful shell limits from the distance histogram
		
		Parameters
		----------
		cut_off_position : {float, 'arena_fraction'}, optional
			Only peaks at larger distances than cut_off_positon are
			considered. This way the first peak that is typically given
			by the size of a grid fiel is cut off.
			If cut_off_position is 'arena_fraction', it is automatically set
			to 0.15 the size of the arena. Since most grid cells have less
			than 6x6 firing fields, this cut off is not too big to miss 
			the relevant peak of the histogram. It does not include the
			second peak, so it is not too large, either.
		
		Returns
		-------
		"""
		self.maximum_index = maximum_index
		if cut_off_position == 'arena_fraction':
			self.cut_off_position = self.get_cut_off_position()
		else:
			self.cut_off_position = cut_off_position
		if shell_limits_choice is not None:
			n, centers = self.get_distancehistogram_and_centers(bins=bins,
																weights=weights)
			self.maxima_positions, self.maxima_values = \
				arrays.get_maximapositions_maximavalues(x=centers, y=n,
				neighborhood_size_as_fraction=neighborhood_size_as_fraction,
				threshold_difference_as_fraction=threshold_difference_as_fraction)

			if not self._sufficient_maxima:
				d = np.mean(dist.pdist(self.pos, 'euclidean'))
				shell_limits = (
					# np.array([0.8, 1.2])
					# * np.mean(dist.pdist(self.pos, 'euclidean'))
					np.array([d - d / 6., d + d / 6.])
				)

			else:
				typical_distance = self.get_typical_distance()
				if shell_limits_choice == 'automatic_single':
					shell_limits = (
						# np.array([0.8, 1.2])
						# * typical_distance
						np.array([typical_distance - typical_distance / 6.,
								  typical_distance + typical_distance / 6.])
					)
					# shell_limits = np.array([0.2, 0.3])
				elif shell_limits_choice == 'automatic_single_for_bands':
					d = typical_distance / 2.
					shell_limits = (
						np.array([d - d / 6.,
								  d + d / 6.])
					)
				elif shell_limits_choice == 'automatic_single_for_quadratic':
					d = 2./3 * typical_distance
					shell_limits = (
						np.array([d - d / 6.,
								  d + d / 6.])
					)
				elif shell_limits_choice == 'automatic_double':
					shell_limits = (
						np.array([[0.8, 1.2], [1.99, 2.001]])
						* typical_distance
					)
				elif shell_limits_choice == 'try_multiple_limits':
					# Try 30 shells. Take best
					gridscores = []
					radius = self._get_radius()
					spacings = np.linspace(radius  / 5., radius, 30)
					for s in spacings:
						shell_limits = np.array([s - s / 6., s + s / 6.])
						gridscore = self.get_gridscore(
							method='given_shell_limits',
							gridscore_norm=None,
							std_threshold=None,
							given_shell_limits=shell_limits
						)
						gridscores.append(gridscore)
					idx_best_limit = np.argmax(gridscores)
					s = spacings[idx_best_limit]
					shell_limits =  np.array([s - s / 6., s + s / 6.])
				elif shell_limits_choice == 'hardcoded_limits':
					shell_limits = np.array([84, 118])

		else:
			shell_limits = None

		print(shell_limits)
		return shell_limits

	def _sufficient_maxima(self, maximum_index, maxima_positions):
		if maximum_index >= len(maxima_positions):
			print('WARNING: Optimal shell size could not be determined '
				  'because insufficient maxima were found. '
				  'Shell size will be set to the mean distance!')
			return False
		else:
			return True

	def _get_radius(self):
		return (self.arena_limits[0, 1] - self.arena_limits[0, 0]) / 2.

	def _neighbor_positions(self, idx):
		neighbor_idx = self._neighbor_indeces(idx)
		return self.pos[neighbor_idx]

	def _psi_n(self, idx, std_threshold=None, compare_to_other_symmetries=False):
		pos = self.pos[idx]
		neighbor_pos = self._neighbor_positions(idx)
		return get_psi_n(pos, neighbor_pos, n=self.n_symmetry,
						 std_threshold=std_threshold,
						 compare_to_other_symmetries=compare_to_other_symmetries)

	def get_number_of_neighbors(self, idx):
		"""
		Returns the number of neighbors of spike with index idx
		"""
		return len(self._neighbor_indeces(idx))

	def get_number_of_neighbors_array(self):
		"""
		Number of neighbors of each spike, in correct indexing order
		"""
		n_neighbors = np.empty(self.n_spikes)
		for idx in np.arange(self.n_spikes):
			n_neighbors[idx] = self.get_number_of_neighbors(idx)
		return n_neighbors

	def get_norm_psinorm(self, normalization):
		"""
		Returns a normalization value and a normalization array

		norm: which is the same for the psi_k value of every spike
		psinorm: which might be different for each individual spike
		
		Parameters
		----------
		normalization : str or None
			None:
			No extra normalization (the psi values are conventionally
			normalized between 0 and 1 in the corresponding function though)

			'all_neighbor_pairs':
			The psi values of a spike are multiplied with the total number
			of spikes and divided by the total number of all occurring
			neighbor pairs. NB: to cancel the conventional normalization of
			psi_k by dividing by the number of neighbors of spike k, we
			multiply with the number of neighbors of spike k (This is done
			with `psinorm`).

		Returns
		-------
		norm : float
		psinorm : ndarray
		"""
		if normalization is None:
			norm = 1
			psinorm = np.ones(self.n_spikes)
		elif normalization == 'all_neighbor_pairs':
			norm = (
				self.n_spikes /
				np.sum(self.get_number_of_neighbors_array())
			)
			psinorm = self.get_number_of_neighbors_array()
		return norm, psinorm

	def psi_n_all(self, n_symmetry=6, shell_limits=None,
				  normalization=None, std_threshold=None,
				  compare_to_other_symmetries=False):
		self.n_symmetry = n_symmetry
		if shell_limits is not None:
			self.shell_limits = np.asarray(shell_limits)
		else:
			self.shell_limits = None
		norm, psinorm = self.get_norm_psinorm(normalization=normalization)
		psi = []
		for idx in np.arange(self.pos.shape[0]):
			psi.append(self._psi_n(idx, std_threshold=std_threshold,
								   compare_to_other_symmetries=compare_to_other_symmetries) * psinorm[idx])
		return np.array(psi) * norm

	def get_distancehistogram_and_centers(self, bins=100, weights=None):
		"""
		Returns number of occurrences at distances given by centers

		Instead of returning the

		Parameters
		----------



		Returns
		-------
		"""
		# r = self.radius
		# diameter = 2 * r
		d = dist.pdist(self.pos, 'euclidean')
		d = np.sort(d)
		if weights == 'inverse_distance':
			weights = 1. / d
		n, bin_edges = np.histogram(d, bins=bins, weights=weights)
		centers = arrays.binedges2centers(bin_edges)
		return n, centers

	# def get_disthist_centers_maxpos_maxval(self,
	# 									   neighborhood_size=20,
	# 									   threshold_difference=0.05):
	# 	#TODO: How to avoid that distance histogram is calculated again
	# 	# in plotting to get the maxima (here I did it by returning everything at once)
	# 	n, centers = self.get_distancehistogram_and_centers()
	# 	n_normalized = n / np.amax(n)
	# 	maxima_boolean = general_utils.arrays.get_local_maxima_boolean(
	# 		n_normalized, neighborhood_size, threshold_difference)
	# 	maxima_positions = centers[maxima_boolean]
	# 	maxima_values = n[maxima_boolean]
	# 	return n, centers, maxima_positions, maxima_values

	def get_typical_distance(self):
		"""
		The typical distance obtained from distance histogram

		The distance histogram has multiple peaks.
		For good grid cells these are clearly separated.

		Parameters
		----------
		maximum_index : int
			The index of the the peak (0 would give the first peak)
		cut_off_position: float
			Peaks at locations smaller than `minimum` are removed.
			Note: First they are removed, then we use the index `n_max`.

		Returns
		-------
		float
		"""
		mp, mv = self.maxima_positions, self.maxima_values
		sorted_max_pos = np.sort(mp[mp > self.cut_off_position])
		try:
			typical_distance = sorted_max_pos[self.maximum_index]
		except IndexError:
			print('WARNING: typical distance could not be determined and'
				  ' is set to NaN')
			typical_distance = np.nan
		return typical_distance

	def get_heatmap(self, bins=101):
		# if not bins:
		# 	bins = self.arena_limits[0, 1] * 2
			# bins = 200
		pos_x = self.pos[:, 0]
		pos_y = self.pos[:, 1]
		heatmap, xedges, yedges = np.histogram2d(
			pos_x, pos_y, bins=bins)
		return heatmap

	def get_ratemap(self, sigma=1):
		heatmap = self.get_heatmap()
		rm = ndimage.filters.gaussian_filter(heatmap, sigma=sigma)
		return rm
