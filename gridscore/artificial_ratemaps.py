import numpy as np
from scipy import stats
import general_utils.misc as misc
import general_utils.arrays as arrays

def get_equidistant_positions(r, n, boxtype='linear', distortion=0., on_boundary=False):
	"""Returns equidistant, symmetrically distributed coordinates

	Works in dimensions higher than One.
	The coordinates are taken such that they don't lie on the boundaries
	of the environment but instead half a lattice constant away on each
	side (now optional).
	Note: In the case of circular boxtype, positions outside the cirlce
		are thrown away.

	Parameters
	----------
	r : array_like
		Dimensions of the box [Rx, Ry, Rz, ...]
		If `boxtype` is 'circular', then r can just be an integer, if it
		is an array the first entry is taken as the radius
	n : array_like
		Array of same shape as r, number of positions along each direction
	boxtype : string
		'linear': A quadratic arrangement of positions is returned
		'circular': A ciruclar arrangement instead
	distortion : float or array_like or string
		If float or array: Maximal length by which each lattice coordinate (x and y separately)
		is shifted randomly (uniformly)
		If string:
			'half_spacing': The distortion is taken as half the distance between
			two points along a perfectly symmetric lattice (along each dimension)
	on_boundary : bool
		If True, positions can also lie on the system boundaries
	Returns
	-------
	(ndarray) of shape (m, len(n)), where m < np.prod(n) for boxtype
	'circular', because points at the edges are thrown away. Note
	len(n) is the dimensionality.
	"""
	if distortion == 'half_spacing':
		distortion = r / (n-1)
	r, n, distortion = np.asarray(r), np.asarray(n), np.asarray(distortion)
	if not on_boundary:
		# Get the distance from the boundaries
		d = 2*r/(2*n)
	else:
		# Set the distance from the boundaries to zero
		d = np.zeros_like(r)
	# Get linspace for each dimension
	spaces = [np.linspace(-ra+da, ra-da, na) for (ra, na, da) in zip(r, n, d)]
	# Get multidimensional meshgrid
	Xs = np.meshgrid(*spaces)
	if boxtype == 'circular':
		distance = np.sqrt(np.sum([x**2 for x in Xs], axis=0))
		# Set grid values outside the circle to NaN. Note: This sets the x
		# and the y component (or higher dimensions) to NaN
		for x in Xs:
			x[distance>r[0]] = np.nan
	# Obtain positions file (shape: (n1*n2*..., dimensions)) from meshgrids
	positions = np.array(zip(*[x.flat for x in Xs]))
	# Remove any subarray which contains at least one NaN
	# You do this by keeping only those that do not contain NaN (negation: ~)
	positions = positions[~np.isnan(positions).any(axis=1)]
	dist = 2*distortion * np.random.random_sample(positions.shape) - distortion
	return positions + dist

class ArtificialRatemaps:
	def __init__(self, distribution='whitenoise',
				 spacing=200, gridspacing=0.15,
				 fieldsize=0.04, orientation=0,
				 shiftnoise=0.2,
				 fieldlocation_noise=0.0,
				 field_factor_noise=0.0,
				 locality_of_rotation_angle=None,
				 local_noise=None):
		"""
		
		
		Parameters
		----------
		distribution : str or list of str
			Typically just one element.
		
		Returns
		-------
		"""
				 # arena_limits=np.array([0, 1], [0, 1])):
		self.field_factor_noise = field_factor_noise
		self.shift = (
			np.array([0.5, 0.5]) +
			(2 * shiftnoise * np.random.random_sample(2) - shiftnoise)
		)
		self.fieldlocation_noise = fieldlocation_noise
		self.distribution = np.atleast_1d(distribution)
		# self.arena_limit = arena_limits
		self.spacing = spacing
		self.gridspacing = gridspacing
		self.orientation = orientation
		self.locality_of_rotation_angle = locality_of_rotation_angle
		self.local_noise = local_noise
		self.fieldcovariance = np.array(
			[
				[fieldsize**2, 0],
				[0, fieldsize**2]
			]
		)
		self.rm_list = self.get_list_of_ratemaps()

	def get_list_of_ratemaps(self):
		"""
		Returns a ratemap for each element in self.distribution
		"""
		rm_dict = dict(
			whitenoise=self.get_whitenoise_rm,
			flat=self.get_flat_rm,
			singlebump=self.get_singlebump_rm,
			grid=self.get_grid_rm,
			manybumps=self.get_manybumps_rm,
			band=self.get_band_rm,
			quadratic=self.get_quadratic_rm,
		)
		rm = []
		for dist in np.atleast_1d(self.distribution):
			rm.append(rm_dict[dist]())
		# rm = rm_dict[self.distribution]()
		return rm

	def get_whitenoise_rm(self):
		rm = np.random.random_sample((self.spacing, self.spacing))
		return rm

	def get_flat_rm(self):
		rm = np.ones((self.spacing, self.spacing))
		return rm

	def get_singlebump_rm(self):
		linspace = np.linspace(0, 1, self.spacing)
		x, y = np.meshgrid(linspace, linspace)
		positions = np.dstack((x, y))
		mean = np.random.random_sample(2)
		# The random variate
		rv = stats.multivariate_normal(mean, self.fieldcovariance)
		rm = rv.pdf(positions)
		return rm

	def get_manybumps_rm(self, n_bumps=24):
		linspace = np.linspace(0, 1, self.spacing)
		x, y = np.meshgrid(linspace, linspace)
		positions = np.dstack((x, y))
		overlap = 0.2
		b = 1 + overlap
		a = -overlap
		field_locations = (b-a) * np.random.random_sample((n_bumps, 2)) + a
		rm = np.zeros((self.spacing, self.spacing))
		for fl in field_locations:
			rm += stats.multivariate_normal(
				fl, self.fieldcovariance).pdf(positions)
		return rm

	def get_grid_rm(self):
		linspace = np.linspace(0, 1, self.spacing)
		x, y = np.meshgrid(linspace, linspace)
		positions = np.dstack((x, y))
		field_locations = self.get_gridfield_locations()
		field_locations = self.add_noise(field_locations,
										 self.fieldlocation_noise,
										 local_noise=self.local_noise)
		rm = np.zeros((self.spacing, self.spacing))
		for fl in field_locations:
			field_factor = 1 + self.field_factor_noise * np.random.random_sample(1)
			rm += field_factor * stats.multivariate_normal(fl, self.fieldcovariance).pdf(positions)
		return rm

	def get_band_rm(self):
		"""
		Currently a hacky method.
		
		We take a perfect grid, which has to be of orientation 0,
		and make it very wide along one dimension.
		
		Parameters
		----------
		
		
		
		Returns
		-------
		"""
		linspace = np.linspace(0, 1, self.spacing)
		x, y = np.meshgrid(linspace, linspace)
		positions = np.dstack((x, y))
		# Take perfect hexagonal grid
		field_locations = self.get_gridfield_locations()
		rm = np.zeros((self.spacing, self.spacing))
		# Make
		# self.fieldcovariance[0,0] = 0.04**2
		self.fieldcovariance[0,0] = 1.0
		for fl in field_locations:
			# field_factor = 1 + self.field_factor_noise * np.random.random_sample(1)
			rm += stats.multivariate_normal(fl, self.fieldcovariance).pdf(positions)
		return rm

	def get_quadratic_rm(self):
		linspace = np.linspace(0, 1, self.spacing)
		x, y = np.meshgrid(linspace, linspace)
		positions = np.dstack((x, y))
		# Get positions on a quadratic grid
		# Note that the function get_equidistant_position places them on a grid
		# from -r to r, so you lose 3/4 of the fields
		# field_locations = get_equidistant_positions(
		# 	r=np.array([1, 1]), n=np.array([10, 10])
		# )

		field_locations = np.array(
			[
				[0., 0./3],
				[0., 1./3],
				[0., 2./3],
				[0., 3./3],

				[1. / 3, 0. / 3],
				[1. / 3, 1. / 3],
				[1. / 3, 2. / 3],
				[1. / 3, 3. / 3],

				[2. / 3, 0. / 3],
				[2. / 3, 1. / 3],
				[2. / 3, 2. / 3],
				[2. / 3, 3. / 3],

				[3. / 3, 0. / 3],
				[3. / 3, 1. / 3],
				[3. / 3, 2. / 3],
				[3. / 3, 3. / 3],

			]
		)
		rm = np.zeros((self.spacing, self.spacing))
		self.fieldcovariance[0,0] = ((1./3) / 6.)**2
		self.fieldcovariance[1,1] = ((1./3) / 6.)**2
		for fl in field_locations:
			# field_factor = 1 + self.field_factor_noise * np.random.random_sample(1)
			rm += stats.multivariate_normal(fl, self.fieldcovariance).pdf(positions)
		return rm

	def add_noise(self, positions, noise=0.0, local_noise=None):
		"""
		Adds symmetric Gaussian noise to the given positions.
		
		NB: When creating artificial grids, the grid field locatoin fall
		way outside the box. You typically have 271 field locations. Ranging
		from negative to positive values. 
		
		Parameters
		----------
		positions : ndarray of shape (N, 2)
		noise : float
			Standard deviation of the Gaussian noise.
		local_noise : None or str
			If None, the noise is applied to all grid fields.
			If 'right_side', noise is only added to grid fields in the
			right half of the arena.
		Returns
		-------
		ret : ndarray of shape (N, 2)
		"""
		sx, sy = positions.shape
		pos_noise = noise * np.random.randn(sx, sy)
		if local_noise == 'right_side':
			pos_noise[positions[:, 0] <= 0.5] = 0.
		return positions + pos_noise

	def get_gridfield_locations(self, sidelength=10):
		"""
		Returns the locations of the centers of the gridfields
		
		NB: currently the arena is quadratic with siderange: [0, 1]
		
		Parameters
		----------
		spacing : float
			Spacing of the grid
		shift : array_like of shape (2)
			Vector by which the locations are shifted.
			NB: the hexagon is by default centered around [0, 0]. Using a
			shift it can be centered around the arena.
		grid_orientation : float
			Angle, in degrees, of the primary axis of the grid
		sidelength : int
			Defines the size of the underlying hexagon.
			The default sidelenght of 10 results in 271 grid fields.
			This should ensure a large enough region, such that there are no
			boundary effects.
		Returns
		-------
		locations : ndarray of shape (N, 2) with
					N = 1 + 3 sidelength (sidelength-1)
		"""
		hexcoords = misc.get_hexagonal_coordinates(sidelength)
		locations = []
		for h in hexcoords:
			locations.append(misc.hex2cart(q=h[0], r=h[1],
										   size=self.gridspacing))
		locations = arrays.rotate_positions(locations, angle=self.orientation,
					locality_of_rotation_angle=self.locality_of_rotation_angle)
		locations = arrays.shift_positions(locations, self.shift)
		return locations

