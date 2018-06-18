import numpy as np
import functools

import scipy.ndimage as ndimage
import scipy.signal as signal
# from pylab import *

import general_utils.arrays
from general_utils.misc import idx2loc
##################################
##########	Gridness	##########
##################################
def get_correlation_2d(a, b, mode='same'):
	"""
	Determines array of Pearson correlation coefficients.

	Array b is shifted with respect to a and for each possible shift
	the Pearson correlation coefficient for the overlapping part of
	the two arrays is determined and written in the output rate.
	For a = b this results in the  Pearson auto-correlogram.
	Note that erratic values (for example seeking the correlation in
	array of only zeros) result in np.nan values.

	UPDATE: The gist of this function is now in pearson_correlate2d,
	it is only kept because it also returns the spacing of the
	correlogram and is thus needed for compatibility.
	NOTE: It has only been tested for mode='same' and 
	shapes of odd length, so better stick to it.

	Parameters
	----------
	a : array_like
		Shape needs to be (N x N), N must be an odd number.
	b : array_like
		Needs to be of same shape as a
	mode : string, optional
		'same': Computes correlogram only for the original scale of the box
				NB: Better use 'same' always. 
		'full': Computes correlogram for twice the boxlenth

	Returns
	-------
	output : ndarray
		Array of Pearson correlation coefficients for shifts of b
		with respect to a.
		The shape of the correlations array is (M, M) where M = 2*N + 1.
	"""
	correlations = pearson_correlate2d(a, b, mode=mode)
	# Set NaNs to zero
	correlations[np.isnan(correlations)] = 0.
	corr_spacing = correlations.shape[0]
	return corr_spacing, correlations


def pearson_correlate2d(in1, in2, mode='same', fft=True):
	"""
	Pearson cross-correlation of two 2-dimensional arrays.

	Cross correlate `in1` and `in2` with output size determined by `mode`.
	NB: `in1` is kept still and `in2` is moved.

	Parameters
	----------
	in1 : array_like
		First input.
	in2 : array_like
		Second input. Should have the same number of dimensions as `in1`.
		If operating in 'valid' mode, either `in1` or `in2` must be
		at least as large as the other in every dimension.
	mode : str {'full', 'valid', 'same'}, optional
		A string indicating the size of the output:
		``full``
		   The output is the full discrete linear cross-correlation
		   of the inputs. (Default)
		``valid``
		   The output consists only of those elements that do not
		   rely on the zero-padding.
		``same``
		   The output is the same size as `in1`, centered
		   with respect to the 'full' output.
	Returns
	-------
	pearson_corr : ndarray
		A 2-dimensional array containing a subset of the discrete pearson
		cross-correlation of `in1` with `in2`.
	"""
	kwargs = dict(mode=mode, fft=fft, normalize=True,
				  set_small_values_zero=1e-10)
	corr = functools.partial(correlate2d, **kwargs)
	ones = np.ones_like(in1)
	pearson_corr = (
		(corr(in1, in2) - corr(ones, in2) * corr(in1, ones))
		/ (
			np.sqrt(corr(in1 ** 2, ones) - corr(in1, ones) ** 2)
			* np.sqrt(corr(ones, in2 ** 2) - corr(ones, in2) ** 2)
		)
	)
	return pearson_corr


def correlate2d(in1, in2, mode, fft, normalize=True,
				set_small_values_zero=None):
	"""
	Correlate two 2-dimensional arrays using FFT and possibly normalize

	NB: `in1` is kept still and `in2` is moved.

	Convenience function. See signal.convolve2d or signal.correlate2d
	for documenation. 
	Parameters
	----------
	normalize : Bool
		Decide wether or not to normalize each element by the
		number of overlapping elements for the associated displacement
	set_small_values_zero : float, optional
		Sometimes very small number occur. In particular FFT can lead to
		very small negative numbers.
		If specified, all entries with absolute value smalle than 
		`set_small_values_zero` will be set to 0.
	Returns
	-------

	"""
	if normalize:
		ones = np.ones_like(in1)
		n = signal.fftconvolve(ones, ones, mode=mode)
	else:
		n = 1
	if fft:
		# Turn the second array to make it a correlation
		ret = signal.fftconvolve(in1, in2[::-1, ::-1], mode=mode) / n
		if set_small_values_zero:
			condition = (np.abs(ret) < set_small_values_zero)
			if condition.any():
				print('Tiny values were set to 0')
				ret[condition] = 0.
		return ret
	else:
		return signal.correlate2d(in1, in2, mode=mode) / n


class Gridness():
	"""
	A class to get information on gridness of correlogram

	Parameters
	----------
	a : ndarray
		A square array containing correlogram data
	radius : float
		The radius of the correlogram
		This should either be 1 or 2 times the radius of the box from
		which the correlogram was obtained. 1 if mode is 'same' 2 if
		mode is 'full'
	neighborhood_size : int
		The area for the filters used to determine local peaks in the
		correlogram.
	threshold_difference : float
		A local maximum needs to fulfill the condition that
	method : string
		Gridness is defined in different ways. `method` selects one
		of the possibilites found in the literature.
	Returns
	-------
	"""

	def __init__(self, a, radius=None, neighborhood_size=5,
				 threshold_difference=0.1, method='Weber',
				 n_contiguous=16, type='hexagonal'):
		self.a = a
		self.radius = radius
		self.method = method
		self.neighborhood_size = neighborhood_size
		self.threshold_difference = threshold_difference
		self.spacing = a.shape[0]
		self.x_space = np.linspace(-self.radius, self.radius, self.spacing)
		self.y_space = np.linspace(-self.radius, self.radius, self.spacing)
		self.X, self.Y = np.meshgrid(self.x_space, self.y_space)
		self.distance = np.sqrt(self.X * self.X + self.Y * self.Y)
		self.distance_1d = np.abs(self.x_space)
		self.type = type
		if len(a.shape) > 1:
			if type == 'hexagonal':
				self.n_peaks = 7
			elif type == 'quadratic':
				self.n_peaks = 5
			if method == 'sargolini' or method == 'langston':
				self.set_labeled_array(n_contiguous, n_peaks=self.n_peaks)
			elif method == 'sargolini_extended':
				self.set_labeled_array(n_contiguous, extended=True,
									   n_peaks=self.n_peaks)
			self.set_center_to_inner_peaks_distances()
			self.set_inner_radius_outer_radius_grid_spacing()

	def set_spacing_and_quality_of_1d_grid(self):
		# We pass a normalized version (value at center = 1) of the correlogram
		maxima_boolean = general_utils.arrays.get_local_maxima_boolean(
			self.a / self.a[self.spacing / 2], self.neighborhood_size,
			self.threshold_difference)
		distances_from_center = np.abs(self.x_space[maxima_boolean])
		# The first maximum of the autocorrelogram gives the grid spacing
		try:
			self.grid_spacing = np.sort(distances_from_center)[1]
		except:
			self.grid_spacing = 0.0
		# The quality is taken as the coefficient of variation of the
		# inter-maxima distances
		# You could define other methods. Use method = ... for this purpose.
		distances_between_peaks = (np.abs(distances_from_center[:-1]
										  - distances_from_center[1:]))
		self.std = np.std(distances_between_peaks)
		self.quality = (np.std(distances_between_peaks)
						/ np.mean(distances_between_peaks))

	def set_labeled_array(self, n_contiguous, extended=False, n_peaks=7):
		"""
		Finds peaks in correlogram and labels them.

		Peaks are clusters (or features) of adjacent pixels (diagonally adjacent
		also counts) with values above threshold.
		Each cluster gets an integer label and each pixel of the same
		cluster gets the same label.
		We keep only the (6+1) most central peaks.
		Everthing else in the array is set to zero.

		Parameters
		----------
		n_contiguous : int
			Only clusters of `n_contiguous` particles or more are kept
			and labeled (except for the central one, which can be arbitrarily
			small)

		Returns
		-------
		Nothing
		"""
		# Definition of neighborhood (diagonal neighborhood matters)
		structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
		clipped_a = self.a.copy()
		# Keep only pixels above threshold value (typically 0.1)
		clipped_a[clipped_a <= self.threshold_difference] = 0.
		# Get all clusters
		labeled_array, self.num_features = ndimage.measurements.label(
			clipped_a, structure=structure)
		# Keep only clusters of size > n_contiguous
		for n in np.arange(1, self.num_features + 1):
			if len(labeled_array[labeled_array == n]) < n_contiguous:
				labeled_array[labeled_array == n] = 0.
		# Keep only the 6+1 most central clusters
		# Note: the number of clusters can get smaller than 7
		labeled_array = self.keep_n_most_central_features(
			labeled_array, n=n_peaks)
		self.labeled_array = labeled_array
		if extended:
			self.labeled_array = self.keep_meaningful_central_features(
				self.labeled_array)

	def get_peak_locations(self, return_as_list=False):
		"""
		Returns the center of mass location of each label.

		This gives the locations of the 7 most central peaks

		Returns
		-------
		Positions : list of tuples
			example: [(x1, y1), (x2, y2), ...]
		"""
		occurring_labels = np.intersect1d(
			self.labeled_array, self.labeled_array)
		location_indices = ndimage.measurements.center_of_mass(
			self.a, labels=self.labeled_array, index=occurring_labels)
		locations = []
		# center_idx = (self.spacing - 1) / 2.
		if return_as_list:
			for idx in location_indices:
				locations.append(
					[idx2loc(idx[0], self.radius, self.spacing),
					 idx2loc(idx[1], self.radius, self.spacing)]
				)
		else:
			for idx in location_indices:
				loc0 = idx2loc(idx[0], self.radius, self.spacing)
				loc1 = idx2loc(idx[1], self.radius, self.spacing)
				# if np.sum([loc0**2, loc1**2]) > 0.01:
				locations.append(
					(loc0, loc1))
		# TODO: Good grid always have seven peaks, but you have an eights location
		# very close to zero. Check where it comes from.
		# good_peak_number = 6
		# if len(locations) != good_peak_number:
		# 	locations = np.ones((good_peak_number, 2)) * np.nan
		return locations

	def remove_location_of_center_peak(self, peak_locations, threshold=0.01):
		"""
		Remove the most central location of all peak location tuples
		"""
		locations = []
		for loc in peak_locations:
			if (loc[0] ** 2 + loc[1] ** 2) < threshold:
				pass
			else:
				locations.append(loc)
		return locations

	def get_grid_axes_angles(self):
		"""
		Returns the angles to three peaks in the correlogram.

		The angles to the peaks closest to 0, 60 and -60 degrees are selected.
		These are the 3 principal axes.
		NB: In the arctan2 function the x and y values are switched on
		purpose.

		Returns
		-------
		List of 3 angles.
		"""
		peak_locations = self.remove_location_of_center_peak(
			self.get_peak_locations())
		if len(peak_locations) < 3:
			return np.asarray([np.nan, np.nan, np.nan])
		# print peak_locations
		angles = []
		for loc in peak_locations:
			# NB: The first argument is y and the second argument is x
			# Since the peak locations originate from the indices,
			# where axis=0 is the arranged from top to bottom and
			# axis=1 is arranged from left to right, this already
			# is a switch of (idx0, idx1) with (-y, x)
			# We don't use -loc[0] though, because when we plot we use
			# a contourplot that corresponds to imshow(origin='lower').
			# We thus have the transform (idx0, idx1) to (y,x)
			angle = np.arctan2(loc[0], loc[1])
			angles.append(angle)
		angles = np.asarray(angles)
		ax1 = general_utils.arrays.find_nearest(angles, 0)
		ax2 = general_utils.arrays.find_nearest(angles, np.pi / 3)
		ax3 = general_utils.arrays.find_nearest(angles, - np.pi / 3)
		return np.asarray([ax1, ax2, ax3])

	def get_sorted_feature_distance_array(self, labeled_array):
		"""
		Returns structured array of labels and distances

		The structured array is sorted with increasing distance from the
		cluster (labeled with `label`) to the coordinate center

		Parameters
		----------
		see labeled_array

		Returns
		-------
		feature_distance_array : ndarray (structured)
		"""
		my_dtype = [('label', int), ('distance', float)]
		feature_distance = []
		# Get all the ocurring labels
		all_labels = np.unique(labeled_array[labeled_array != 0])
		for l in all_labels:
			d = self.distance[labeled_array == l]
			feature_distance.append(
				(l, np.mean(d))
			)
		feature_distance_arr = np.array(feature_distance, my_dtype)
		feature_distance_arr.sort(order='distance')
		return feature_distance_arr

	def keep_n_most_central_features(self, labeled_array, n=7):
		"""
		Keeps only the n most central features

		Parameters
		----------
		labeled_array : ndarray
			Array with labeled features
		n : int
			See above

		Returns
		-------
		reduced labeled_array : ndarray
			Array of same shape as labeled_array, where at most
			the 6+1 most central clusters are still labeled
		"""
		feature_distance_arr = self.get_sorted_feature_distance_array(
			labeled_array)
		labels_to_drop = feature_distance_arr['label'][n:]
		for l in labels_to_drop:
			labeled_array[labeled_array == l] = 0
		# return self.keep_meaningful_central_features(labeled_array)
		return labeled_array

	def keep_meaningful_central_features(self, labeled_array):
		"""
		Removes clusters that are too far outside

		For bad grids there are typically not six inner peaks around the
		center. If we keep the 6+1 most central peaks (as done in
		keep_n_most_central_peaks) this might lead to peaks of very different
		distances to the center. This does not make sense, because then the
		outer radius is drawn around the largest outlier. In this scenario
		we instead would like to draw the outer radius around the furthest
		outside inner peaks, although this leads to less than 6+1 peaks
		considered. We identify this scenario by comparing the distance of the
		outermost peak to the center with the distance of one of the inner
		peaks to the center (actually the third one, which should exist).
		If the outermost peak is more than 1.5 further away than the inner
		one it is rejected.
		This is not mentioned in Sargolini 2006, but it is desirable still.
		Note: You neeed to run keep_n_most_central_features first.

		Returns
		-------
		reduced labeled_array : ndarray
		"""
		feature_distance_arr = self.get_sorted_feature_distance_array(
			labeled_array)
		labels_all = feature_distance_arr['label']
		try:
			while (feature_distance_arr['distance'][-1]
					   > 1.5 * feature_distance_arr['distance'][2]):
				feature_distance_arr = np.delete(feature_distance_arr, -1,
												 axis=0)
		except IndexError:
			pass
		labels_to_keep = feature_distance_arr['label']
		labels_to_drop = np.setdiff1d(labels_all, labels_to_keep)
		for l in labels_to_drop:
			labeled_array[labeled_array == l] = 0
		return labeled_array

	def get_peak_center_distances(self, n):
		"""
		Distance to center of n most central peaks

		The center itself (distance 0.0) is excluded

		Parameters
		----------
		n : int
			Only the `n` most central peaks are returned

		Returns
		-------
		output : ndarray
			Sorted distances of `n` most central peaks.

		"""
		maxima_boolean = general_utils.arrays.get_local_maxima_boolean(
			self.a, self.neighborhood_size, self.threshold_difference)
		first_distances = np.sort(self.distance[maxima_boolean])[1:n + 1]
		return first_distances

	def get_central_cluster_bool(self):
		"""
		Returns the array with True at the location of the center peak pixels
		"""
		idx_central = int((self.spacing - 1) / 2.)
		central_label = self.labeled_array[idx_central][idx_central]
		central_cluster_bool = (self.labeled_array == central_label)
		# Check if central cluster bool is True everywhere
		# If so, this means that the central cluster is considered to be
		# the entire box. If this occurs we manually say that the central
		# cluster is one off set from the absolute center.
		# This leads to a very small inner radius.
		# Note the double negation in the condition!
		if np.count_nonzero(~central_cluster_bool) == 0:
			central_cluster_bool = ~central_cluster_bool
			central_cluster_bool[int((self.spacing - 2) / 2.)][
				idx_central] = True
		return central_cluster_bool

	def get_inner_radius(self):
		"""
		Returns the inner radius for the correlogram cropping

		For method = 'sargolini', this is the outermost pixel of all the
		pixels of the most central peak

		For method = 'Weber', this is the mean of the distances to the
		first six peaks (excluding the center peak)

		Returns
		-------
		inner_radius : float
		"""
		# If the central cluster basically fills the entire arena, we set it
		# to a high value. If we would set it to the size of the arena,
		# the outer radius could not be largre and we would get meaningless
		# arrays. This is just error handling.
		largest_inner_radius_allowed = 0.8 * self.radius
		if (self.method == 'sargolini'
			or self.method == 'sargolini_extended'
			or self.method == 'langston'):
			central_cluster_bool = self.get_central_cluster_bool()
			r = np.amax(self.distance[central_cluster_bool])
			if r > largest_inner_radius_allowed:
				r = largest_inner_radius_allowed
			return r
		elif self.method == 'Weber':
			first_distances = self.center_to_inner_peaks_distances
			return 0.5 * np.mean(first_distances)

	def get_outer_radius(self):
		"""
		Returns the outer radius for the correlogram cropping

		For method = 'sargolini', this is the outermost pixel of the seven
		most central pixel clusters. If there are less than seven, than the
		outermost pixel of the remaing clusters is taken.

		For method = 'Weber', this is the maximal distances to the 6 closest
		non-center clusters + the inner radius.

		Returns
		-------
		outer_radius : float
		"""
		if (self.method == 'sargolini'
			or self.method == 'sargolini_extended'
			or self.method == 'langston'):
			valid_cluster_bool = np.nonzero(self.labeled_array)
			try:
				return np.amax(self.distance[valid_cluster_bool])
			except ValueError as e:
				print(e)
				print('number of nonzero values in labeled array:')
				print(np.count_nonzero(self.labeled_array))
				return self.radius
		elif self.method == 'Weber':
			first_distances = self.center_to_inner_peaks_distances
			return max(first_distances) + 1.0 * self.inner_radius

	def get_grid_spacing(self):
		"""
		Returns the grid spacing
		"""
		first_distances = self.center_to_inner_peaks_distances
		# If set_center_to_inner_peaks_distances cause an exception
		# because no clusters were found it sets
		# first_distances[1] = self.radius. Here we check for that.
		# set_center_to_inner_peaks_distances
		if first_distances[1] == self.radius:
			return np.nan
		else:
			return np.mean(first_distances)

	def set_inner_radius_outer_radius_grid_spacing(self):
		self.inner_radius = self.get_inner_radius()
		self.outer_radius = self.get_outer_radius()
		self.grid_spacing = self.get_grid_spacing()

	def set_center_to_inner_peaks_distances(self):
		"""
		The distance to the 6 most central peaks to the center
		"""
		first_distances = self.get_peak_center_distances(self.n_peaks - 1)
		if len(first_distances) > 1:
			closest_distance = first_distances[0]
			# We don't want distances outside 1.5*closest distance
			first_distances = [d for d in first_distances if
							   d < 1.5 * closest_distance]
		# If before or after taking the inner most peaks, we only have one
		# or two peaks left, we set them artificially
		if len(first_distances) <= 1:
			first_distances = [0.01, self.radius]
		self.center_to_inner_peaks_distances = first_distances

	def get_cropped_flattened_arrays(self, arrays):
		"""
		Crop arrays, keep only values where inner_radius<distance<outer_radius

		Parameters
		----------
		arrays : ndarray
			Array of shape (n, N, N) with `n` different array of shape `N`x`N`

		Returns
		-------
		output : ndarray
			Array of cropped and flattened arrays for further processing
		"""
		cropped_flattened_arrays = []
		for a in arrays:
			index_i = self.distance < self.inner_radius
			index_o = self.distance > self.outer_radius
			# Set value outside the ring to nan
			a[np.logical_or(index_i, index_o)] = np.nan
			cfa = a.flatten()
			# Keep only finite, i.e. not nan values
			cfa = cfa[np.isfinite(cfa)]
			cropped_flattened_arrays.append(cfa)
		return np.array(cropped_flattened_arrays)

	def get_correlation_vs_angle(self, angles=np.arange(0, 180, 2)):
		"""
		Pearson correlation coefficient (PCC) for unrotated and rotated array.

		The PCC is determined from the unrotated array, compared with arrays
		rotated for every angle in `angles`.

		Parameters
		----------
		angles : ndarray
			Angle values

		Returns
		-------
		output : ndarray tuple
			`angles` and corresponding PCCs
		"""
		# Unrotated array
		a0 = self.get_cropped_flattened_arrays([self.a.copy()])[0]
		rotated_arrays = general_utils.arrays.get_rotated_arrays(self.a, angles)
		cropped_rotated_arrays = self.get_cropped_flattened_arrays(
			rotated_arrays)
		correlations = []
		for cra in cropped_rotated_arrays:
			correlations.append(np.corrcoef(a0, cra)[0, 1])
		return angles, correlations

	def get_grid_score(self, comparison=None):
		"""
		Determine the grid score.

		Thus function determines the grid score by rotating a doughnut
		defined by inner_radius and outer_radius against an unrotated copy.
		It either takes a single doughnut (e.g. for method 'sargolini') or
		tries several doughnuts (e.g. for method 'langston).
		If multiple doughnuts are tried, the one with the highest resulting
		grid score is taken.

		Parameters
		----------
		comparison : str {None, 'mean'} (optional)
			Comparison method between correlations at good
			and bad degree values.
			Default None corresponds to taking the conservative measure
			of min(correlation at good values) - max(correlation at bad values)
			'mean' corresponds to taking the difference of the mean of both.

		Returns
		-------
		max_gs : float
			The grid score corresponding to the given method.
		"""
		good_angles, bad_angles = self.select_angles(self.type)

		# The sargolini grid score determines one outer radius
		# So the list contains just a single element
		outer_radii = np.array([self.outer_radius])
		# The langston grid score tries many outer radii
		if self.method == 'langston':
			# We vary the radius between 1.5 the inner radius and the
			# radius to an edge of the box, i.e., radius * np.sqrt(2)
			steps = 50
			outer_radii = np.linspace(
				1.5 * self.inner_radius,
				self.radius*np.sqrt(
					2), steps)

		gridscores = []
		for outer_radius in outer_radii:
			self.outer_radius = outer_radius
			correlation_good = self.get_correlation_vs_angle(
				angles=np.asarray(good_angles))[1]
			correlation_bad = self.get_correlation_vs_angle(
				angles=np.asarray(bad_angles))[1]
			if comparison == 'mean':
				gridscore = np.mean(correlation_good) - np.mean(correlation_bad)
			else:
				gridscore = min(correlation_good) - max(correlation_bad)

			gridscores.append(gridscore)

		# try:
		max_gs_idx = np.nanargmax(gridscores)
		# except ValueError:
		# 	max_gs_idx = 0

		# Set outer_radius to the value that leads to the highest grid score
		# This is only important for later plotting.
		self.outer_radius = outer_radii[max_gs_idx]
		print('inner radius: ', self.inner_radius)
		print('outer radius: ', self.outer_radius)
		# Return the maximum grid score that was obtained trying all radii
		max_gs = np.amax(gridscores)
		return max_gs

	@staticmethod
	def select_angles(t='hexagonal'):
		"""
		Returns the angles at which correlation should be high or low

		Parameters
		----------
		t : str {'hexagonal', 'quadratic'}
			Type of the structure we are looking for.
		Returns
		-------
		good_angles, bad_angles : tuple of lists
			good_angles are the angles at which correlations should be high
			bad_angles are the angles at which correlations should be low
		"""
		if t == 'hexagonal':
			good_angles = [60, 120]
			bad_angles = [30, 90, 150]
		elif t == 'quadratic':
			good_angles = [90]
			bad_angles = [45, 135]
		return good_angles, bad_angles
