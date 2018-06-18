__author__ = 'simonweber'
import numpy as np
import itertools

def approx_equal(x, y, tolerance=0.001):
	return abs(x - y) <= 0.5 * tolerance * (abs(x) + abs(y))

def cart2pol(x, y):
	"""
	Cartesian to polar coordinates
	"""
	theta = np.arctan2(y, x)
	rho = np.hypot(x, y)
	return np.array([rho, theta])


def pol2cart(rho, phi):
	"""
	Polar to cartesian coordinates
	"""
	x = rho * np.cos(phi)
	y = rho * np.sin(phi)
	return np.array([x, y])


def hex2cart(q, r, size):
	"""
	Convert hexagonal coordinate in axial representation to cartesian
	
	Parameters
	----------
	q : float
		'Column' value of hexagonal coordinate
	r : float
		'Row' value of hexagonal coordinate
	size : float
		Distance between two neighboring coordinates
		
	Returns
	-------
	x, y : tuple of cartesian coordinates
	"""
	# x = size * np.sqrt(3) * (q + r / 2.)
	# y = size * 3. / 2 * r
	x = size * (q + r / 2.)
	y = - size * r * 3 / (2 * np.sqrt(3))
	return x, y

def get_hexagonal_coordinates(sidelength):
	"""
	Returns axial coordinates of hexagon of given sidelength
	
	See http://www.redblobgames.com/grids/hexagons/#basics for
	an illustration of the axial coordinate system
	
	Parameters
	----------
	sidelength : int
	
	
	Returns
	-------
	coords : ndarray of shape (N, 2)
		where N = 1 + 3 sidelength (sidelength-1)
	"""
	a = np.arange(-sidelength+1, sidelength)
	coords = []
	for x, y, z in itertools.product(a, repeat=3):
		if x + y + z == 0:
			coords.append([x, y])
	return np.array(coords)

def idx2loc(idx, radius, nbins):
	"""
	Transforms an index to a location

	Indeces range from 0 to nbins-1
	Locations range from -radius to radius

	Parameters
	----------
	idx : int or float
		An index of an array. Can be from interval: [0, nbins-1].
		If it is a float, the float is floored first.
	radius : float
	nbins : int
		The number of spins in which the space is divided.
	Returns
	-------
	"""
	idx = np.floor(idx)
	return 2 * radius * (idx / (nbins - 1)) - radius

