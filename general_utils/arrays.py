

import numpy as np
import scipy.ndimage as ndimage
import scipy.spatial as spatial

def binedges2centers(binedges):
	"""
	Returns centers of histogram bars from bin edges

	Parameters
	----------
	binedges : (N) ndarray

	Returns
	-------
	centers : (N-1) ndarray
	"""
	centers = (binedges[:-1] + binedges[1:]) / 2
	return centers

def find_nearest(a, value, ret='element', axis=None):
	"""
	Returns the array element that is nearest to the given value

	Parameters
	----------
	a : ndarray
		Array of elements
	value : float
		Target value to which the elements should be compared
	axis : int, optional
		Determines along which axis the nearest entries should be searched
		for.
	ret : strgi
		'index' or 'element'
		NB: `index` is currenlty only used for one dimensional arrays
	Returns
	-------
	"""
	# Transform such that we have to find the values closest to zero
	a_zero = np.abs(a - value)
	idx = a_zero.argmin(axis=axis)
	if ret == 'element':
		if axis is None:
			r = np.ravel(a)[idx]
		# elif axis == 1:
		# 	r = a[np.arange(0, a.shape[0]), idx]
		# elif axis == 0:
		# 	r = a[idx, np.arange(0, a.shape[1])]
		else:
			r = apply_mask(a, idx, axis)
	elif ret == 'index':
		r = idx
	return r

def apply_mask(a, indices, axis):
	"""
	Conevenience function.

	Copied from:
	http://stackoverflow.com/questions/
	15469302/numpy-3d-to-2d-transformation-based-on-2d-mask-array

	Parameters
	----------
	a : ndarray
	indices : ndarray
		Indices along all axes except for `axis`
	axis : int
		The one axis that is missing in indices

	Returns
	-------
	ret : ndarray
		Entries of *a* that are selected with indices along `axis`.
	"""
	magic_index = np.ogrid[tuple(slice(i) for i in indices.shape)]
	magic_index.insert(axis, indices)
	return a[magic_index]

def custom_order_for_some_elements(a, names):
	"""Puts specific elements to beginning of list, if they are present.

	Parameters
	----------
	a : list
		Input list
	names : list
		Element names in desired order. They will be put to the beginning
		of the list in this order.
		This list can be smaller or equal to `a`
	Returns
	-------
	out : list
		Same list as input but with new order

	"""
	# Take reversed list, to ensure the order
	for n in names[::-1]:
		# Do nothing if name is not in the list
		if n in a:
			i = a.index(n)
			a.insert(0, a.pop(i))
	return a


def get_distances(sidelength):
	"""Gets all interpoint distances of array

	BETA!
	Note: Currently it only does so for arrays with 0, 1, 2, 3 spacing

	Parameters
	----------
	sidelength

	Returns
	-------
	"""
	N = sidelength**2
	x_space = np.arange(sidelength)
	y_space = np.arange(sidelength)
	X1, Y1 = np.meshgrid(x_space, y_space)
	distances = np.empty((N, sidelength, sidelength))
	for i in np.arange(N):
		print(i)
		X, Y = np.meshgrid(	x_space - X1.T.reshape(N)[i],
						y_space - Y1.T.reshape(N)[i])
		# Periodic boundary conditions
		X[X > sidelength/2] = sidelength - X[X > sidelength/2]
		X[X < -sidelength/2] = sidelength + X[X < -sidelength/2]
		Y[Y > sidelength/2] = sidelength - Y[Y > sidelength/2]
		Y[Y < -sidelength/2] = sidelength + Y[Y < -sidelength/2]
		distances[i] = np.sqrt(X*X + Y*Y)
	X2 = X1.reshape(N)
	Y2 = Y1.reshape(N)
	def distance_between_neurons(neuron1, neuron2):
		return distances[neuron1][X2[neuron2], Y2[neuron2]]
	dist = np.empty((N, N))
	for i in np.arange(N):
		print(i)
		for j in np.arange(N):
			dist[i, j] = distance_between_neurons(i,j)
	return dist

def take_every_nth(array, n):
	"""
	Returns an array containing the first and every subsequent
	n-th array of the input array

	--------
	Arguments:
	- array: numpy array
	- n: step size for sparsification

	--------
	Returns:
	- The sparsified array
	"""
	array_length = len(array)
	index_array = np.zeros(array_length)
	for i in np.arange(array_length):
		if i % n == 0:
			index_array[i] = 1
	index_array = index_array.astype(bool)
	return array[index_array]

def sparsify_two_dim_array_along_axis_1(array, n):
	"""
	Returns sparsified version of input array.

	-------
	Arguments:
	- array: numpy array of shape (n1, n2)
	- n: step size for sparsification

	--------
	Returns:
	- array, sparsified along axis 1

	--------
	Example:
	- array = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
	- n = 2
	- return: [[1, 3], [5, 7], [9, 11]]
	"""
	s = array.shape
	index_array = np.zeros(s[1])
	for i in np.arange(s[1]):
		if i % n == 0:
			index_array[i] = 1
	index_array = index_array.astype(bool)
	index_array = np.tile(index_array, s[0])
	index_array = index_array.reshape(s[0], s[1])
	a = array[index_array]
	l = len(a) // s[0]
	return a.reshape(s[0], l)


def get_local_maxima_boolean(a, s, threshold_difference):
	"""
	Local maxima of array.

	Parameters
	----------
	a : ndarray
	s : int
		Area (in pixels) of the filters which determine the local maxima.
		Note that the shape is automatically (s, s, ...) matching the
		dimensions of `a`.
	threshold_difference : float
		Local maxima are only taken where the difference
		(local_max - local_min) exceeds `threshold_difference`

	Returns
	-------
	output : ndarray
		Returns a boolean array of same shape as `a` with True only at
		positions of local maxima

	"""

	# maximum_filter sets each point in area (neighborhood X neighborhood)
	# to the maximal value occurring in this area
	data_max = ndimage.filters.maximum_filter(a, s)
	# Create boolean array with True entries at the positions of the local
	# maxima:
	maxima = (a == data_max)
	# Find also local minima and remove maxima that are not distinct enough
	# from the maxima array by setting them False
	data_min = ndimage.filters.minimum_filter(a, s)
	below_threshold = ((data_max - data_min) <= threshold_difference)
	maxima[below_threshold] = False
	return maxima

def get_maximapositions_maximavalues(
		x, y, neighborhood_size_in_percent, threshold_difference_in_percent):
	"""
	Return maxima positions and maxima values of function y = f(x)

	Parameters
	----------
	neighborhood_size : float in [0,1]
		Compare get_local_maxima_boolean
	threshold_difference_in_percent : float in [0, 1]
		Compare to get_local_maxima_boolean

	Returns
	-------
	maxima_positions : ndarray
	maxima_values : ndarray
	"""
	y_normalized = y / np.amax(y)
	neighborhood_size = neighborhood_size_in_percent * len(y)
	maxima_boolean = get_local_maxima_boolean(
		y_normalized, neighborhood_size,
		threshold_difference_in_percent)
	maxima_positions = x[maxima_boolean]
	maxima_values = y[maxima_boolean]
	return maxima_positions, maxima_values

def get_mean_inter_peak_distance(a, boxlength, s, threshold_difference):
	"""
	Returns the mean distance between neighboring peaks in data array

	We typically use a = output_rates (for a single frame).

	Parameters
	----------
	a : ndarray (shape N)
	boxlength : float
		Length (in some space) of this array
	s : int
		neighborhood size
		See definition of get_local_maxima_boolean
		Typical value: s = 5
	threshold_difference : float
		See definition of get_local_maxima_boolean
		Typical value: threshold_difference = 0.1


	Returns
	-------
	mean_inter_peak_distance : float
	"""
	maxima_boolean = get_local_maxima_boolean(a, s, threshold_difference)
	x_space = np.linspace(-boxlength/2., boxlength/2.,
						  maxima_boolean.shape[0])
	peak_positions = x_space[maxima_boolean]
	inter_peak_distances = (np.abs(peak_positions[:-1]
									- peak_positions[1:]))
	return np.mean(inter_peak_distances)

def get_rotated_arrays(a, angles):
	"""
	Rotate arrays for each angle in `angles`

	Parameters
	----------
	a : ndarray
		Array to be rotated
	angles : ndarray
		Array of angles

	Returns
	-------
	output : ndarray
		Array of shape len(angles) X a.shape, i.e. one array for each
		rotation angle.

	"""
	rotated_a = np.empty((len(angles), a.shape[0], a.shape[1]))
	for n, angle in enumerate(angles):
		rotated_a[n] = ndimage.interpolation.rotate(a, angle, reshape=False)
	return rotated_a

def get_scaled_array(a, minimum=0, maximum=1):
	"""
	Scales an array linearly, such that it lies between minimum and maximum

	Parameters
	----------
	a : ndarray
	minimum : float
	maximum : float

	Returns
	-------
	ndarray of same shape as `a`
	"""
	min_a, max_a = np.amin(a), np.amax(a)
	return (maximum - minimum) * (a - min_a) / (max_a - min_a) + minimum


def nonzero_and_zero_elements_to_constant(a, c_zero=0, c_nonzero=1):
	"""
	Returns array whose nonzero and zero elements are set to a constant value

	Parameters
	----------
	a : ndarray
	c_zero : float
		Value to which the zero elements are set
	c_nonzero : float other than 0
		Value to which the nonzero elments are set

	Returns
	-------
	ndarray
	"""
	a[a != 0] = c_nonzero
	a[a == 0] = c_zero
	return a


def shift_positions(positions, shift):
	"""
	Shift all positions by shift vector

	Parameters
	----------
	positions : array_like of shape (N, 2)
	shift : array_like of shape (2)

	Returns
	-------
	shifted_pos : ndarray of shape (N, 2)
	"""
	positions = np.asarray(positions)
	shift = np.asarray(shift)
	shifted_pos = positions + shift
	return shifted_pos

def rotate_positions(positions, angle, locality_of_rotation_angle=None):
	"""
	Rotate all positions
	NB: Currently only works for 2 dimensional positions
	See test_rotate_positions

	Parameters
	----------
	positions : array_like of shape (N, 2)
	angle : float
		Angle in degrees.

	Returns
	-------
	rotated_pos : ndarray of shape (N, 2)
	"""
	positions = np.asarray(positions)
	theta = np.radians(angle)
	if not locality_of_rotation_angle:
		c, s = np.cos(theta), np.sin(theta)
		rot = np.array(
			[
				[c, -s],
				[s, c]
			]
		)
		# Note that the j-th element of rotating the i-th vector, x_ij,
		# is given by x_ij = rot_jk positions_ik
		# using Einstein sum convention
		rotated_pos = np.einsum('jk,ik', rot, positions)
	elif locality_of_rotation_angle == 'minus_to_plus_theta_along_y':
		# Note that ymax is a field far outside the box.
		# Rotation of theta at the outermost field on top
		# Rotation of zero at the outermost field on bottom
		rotated_pos = []
		ymax = np.amax(positions[:, 1])
		ymin = np.amin(positions[:, 0])
		for pos in positions:
			phi = theta * ((2 * (pos[1] - ymin) / (ymax - ymin)) - 1)
			c, s = np.cos(phi), np.sin(phi)
			x = c*pos[0] - s*pos[1]
			y = s*pos[0] + c*pos[1]
			rotated_pos.append([x, y])
		rotated_pos = np.asarray(rotated_pos)
	return rotated_pos

def circular_center_of_mass(a, angles):
	"""
	Returns the center of mass of a circular function

	See the supplement of Doeller et al. 2010 for a description

	Parameters
	----------
	a : array_like
		For example firing rates over angles
	angles : array_like of same shape as *a*
		Angles (in radians) over which *a* is defined

	Returns
	-------
	circular center of mass : float
	"""
	a = np.asarray(a) / np.sum(a)
	angles = np.asarray(angles)
	s = np.dot(a, np.sin(angles))
	c = np.dot(a, np.cos(angles))
	return np.arctan2(s, c)

def remove_nan_rows(a):
	"""
	Returns array without rows that contain at least one NaN.

	Parameters
	----------
	a : array_like of shape (M, N)

	Returns
	-------
	ret : ndarray of shape (K, N) with K <= M
	"""
	return a[not_nan_rows_bool(a)]

def not_nan_rows_bool(a):
	"""
	Returns boolian array with True values where row contains no NaNs.

	Parameters
	----------
	a : array_like of shape (M, N)

	Returns
	-------
	ret : boolian array of shape (M,)
	"""
	return ~np.isnan(a).any(axis=1)

def positions_within_circle(positions, origin, radius, ret='positions'):
	distances_from_origin = spatial.distance.cdist(positions, np.array([
		origin]))
	idx = np.argwhere(distances_from_origin[:, 0] <= radius)[:, 0]
	if ret == 'index':
		return idx
	else:
		return positions[idx]

def concatenated_array_from_list_of_arrays(array_list):
	return np.concatenate(tuple(array_list))