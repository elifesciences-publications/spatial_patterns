import numpy as np
import scipy.ndimage as ndimage
import scipy.spatial.distance as dist
from . import arrays
from gridscore import correlogram
from gridscore.spikes import Spikes
from gridscore.plotting import Plot
from gridscore.spikes import get_psi_n
import matplotlib as mpl
import matplotlib.pyplot as plt
from gridscore.arrays import get_maximapositions_maximavalues
from gridscore.plotting import simpleaxis

def positions_from_ratemap(ratemap, arena_limits):
	"""
	Returns the positions of all values in the ratemaps
	
	See also the test.
	
	Parameters
	----------
	ratemap : ndarray of shape (Nx, Ny)
		For quadratic arenas, the ratemap it typically quadratic: Nx = Ny
	arena_limits : ndarray of shape (2, 2)
		The arena limits in x and y direction
	
	Returns
	-------
	positions : ndarray of shape (Nx*Ny, 2)
	"""
	rm_shape = ratemap.shape
	x_length = arena_limits[0, 1] - arena_limits[0, 0]
	y_length = arena_limits[1, 1] - arena_limits[1, 0]
	x_offset = (x_length / rm_shape[0]) / 2.
	y_offset = (y_length / rm_shape[1]) / 2.
	x = np.linspace(arena_limits[0, 0] + x_offset,
					arena_limits[0, 1] - x_offset,
					rm_shape[0])
	y = np.linspace(arena_limits[1, 0] + y_offset,
					arena_limits[1, 1] - y_offset,
					rm_shape[1])
	return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])

def position_with_firing_rate(ratemap, arena_limits):
	positions = positions_from_ratemap(ratemap, arena_limits)
	pos_rate = np.array(
		[
			([0., 0.], r) for r in ratemap.flatten(order='F')
		],
		dtype=[('position', 'float64', 2), ('rate', 'float64')]
	)
	pos_rate['position'] = positions
	return pos_rate

class RateMaps(Spikes, Plot):
	"""
	Class that extracts gridscore information from a firing rate map

	Parameters
	----------
	rm : ndarray of shape (Nx, Ny)
		Firing rate map
	"""
	def __init__(self, rm, arena_limits, test_mode=False):
		if test_mode:
			self.rm = np.load('/Users/simonweber/programming/workspace/gridscore/'
							  'test_ratemap.npy')
			self.arena_limits = np.array([[-50, 50], [-50, 50]])
		else:
			self.rm = rm
			self.arena_limits = arena_limits

		self.n_symmetry = 6
		self.pos_rate = position_with_firing_rate(self.rm, self.arena_limits)
		self.pos = self.pos_rate['position']
		self.rate = self.pos_rate['rate']

		self.cmap = getattr(mpl.cm, 'viridis')

	def get_distances(self):
		return dist.pdist(self.pos, 'euclidean')

	def get_rate_products(self):
		product_array = np.outer(self.rate, self.rate)
		# Get indeces into upper triangle without diagonal
		upper_triangle_idx = np.triu_indices_from(product_array, k=1)
		return product_array[upper_triangle_idx]

	def get_distancehistogram_and_centers(self, bins=100, weights=None):
		d = self.get_distances()
		sort_idx = np.argsort(d)
		d = d[sort_idx]
		rate_products = self.get_rate_products()[sort_idx]
		n, bin_edges = np.histogram(d, bins=bins, weights=rate_products)
		centers = arrays.binedges2centers(bin_edges)
		return n, centers

	def plot_ratemap(self):
		plt.imshow(self.rm, cmap=self.cmap)

	def plot_distancehistogram(self, bins=100):
		n, centers = self.get_distancehistogram_and_centers(bins=bins)
		width = centers[1] - centers[0]
		plt.bar(centers, n, align='center', width=width, color='black')

		maxima_positions, maxima_values = \
			get_maximapositions_maximavalues(
				x=centers, y=n, neighborhood_size_as_fraction=0.1,
			threshold_difference_as_fraction=0.1)

		plt.plot(maxima_positions, maxima_values,
				 marker='o', color='gray', linestyle='none',
				 markersize=5)

		shell_limits = self.get_shell_limits(
			shell_limits_choice='automatic_single',
			cut_off_position=0.1, maximum_index=0, weights=None)

		self.indicate_typical_distance(shell_limits)

		ax = plt.gca()
		plt.setp(ax,
				 xlabel='Distance',
				 ylabel='# pairs',
		)
		ax.locator_params(axis='y', tight=True, nbins=5)
		ax.locator_params(axis='x', tight=True, nbins=5)
		simpleaxis(ax)
		plt.margins(0.1)

	def psi_n_all(self, n_symmetry=6, shell_limits=None):
		self.n_symmetry = n_symmetry
		if shell_limits is not None:
			self.shell_limits = np.asarray(shell_limits)
		else:
			self.shell_limits = None
		psi = []
		for idx in np.arange(self.pos.shape[0]):
			psi.append(
				self._psi_n(idx, std_threshold=None,
							compare_to_other_symmetries=True)
			)
		return np.array(psi)

