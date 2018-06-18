__author__ = 'simonweber'
import unittest
import numpy as np
from .. import ratemaps

class TestSpikes(unittest.TestCase):
	def setUp(self):
		pass
		# self.rm = np.array(
		# 	[
		# 		[2., 5.],
		# 		[3., 0.],
		# 		[1., 0.5]
		# 	]
		# )
		# self.arena_limits = np.array([[0, 1], [0, 1]])

	def test_positions_from_ratemap(self):
		"""
		NB: We treat the first coordinate as the x coordinate, so rows are 
		along the horizontal direction!
		"""
		# The limits of the arena
		arena_limits = np.array([[-0.5, 0.5], [0.0, 2.0]])
		# A 2 x 3 ratemap
		ratemap = np.array([
			[1, 7, 2.],
			[5, 3, 5.],
		])
		# The position of the bins of the ratemap are not at the boundaries,
		# instead they are moved from the boundary by (length / n_bins) / 2.
		# So for the x direction: (1 / 2) / 2 = 1/4. So the leftmost bin is
		# at -0.5 + 0.25.
		# And for the y direction: (2 / 3) / 2 = 1/3. So the lower most bin
		# is at 0 + 1/3.
		expected = np.array([
			[-0.25, 0.0 + 1/3.],
			[ 0.25, 0.0 + 1/3.],
			[-0.25, 1.0],
			[ 0.25, 1.0],
			[-0.25, 2.0 - 1/3.],
			[ 0.25, 2.0 - 1/3.],
		])
		# self.spikes.shell_limits = None
		# result = self.spikes._neighbor_indeces(idx)
		result = ratemaps.positions_from_ratemap(ratemap, arena_limits)
		np.testing.assert_array_equal(expected, result)

	def test_position_with_firing_rates(self):
		# The limits of the arena
		arena_limits = np.array([[-0.5, 0.5], [0.0, 2.0]])
		# A 2 x 3 ratemap
		ratemap = np.array([
			[1, 7, 2.],
			[5, 3, 6.],
		])
		expected = np.array(
			[
				([-0.25, 1/3.], 1),
				([ 0.25, 1/3.], 5),
				([-0.25, 1.0],  7),
				([ 0.25, 1.0],  3),
				([-0.25, 5/3.], 2.),
				([ 0.25, 5/3.], 6.),

			],
		dtype=[('position', 'float64', 2), ('rate', 'float64')]
		)
		result = ratemaps.position_with_firing_rate(ratemap, arena_limits)
		np.testing.assert_array_equal(expected, result)

	def test_get_distances(self):
		rm = np.array([
			[1, 7.],
			[0, 3.],
		])
		arena_limits = np.array([[0, 1], [0, 1]])
		rmps = ratemaps.RateMaps(rm, arena_limits)
		# The positions will be
		# [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]
		sqrt_half = np.sqrt(1/2)
		expected = np.array([0.5, 0.5, sqrt_half, sqrt_half, 0.5, 0.5])
		result = rmps.get_distances()
		np.testing.assert_array_almost_equal(expected, result, decimal=8)

	def test_get_rate_products(self):
		rm = np.array([
			[1, 7.],
			[0, 3.],
			[2., 4.]
		])
		arena_limits = np.array([[0, 1], [0, 1]])
		rmps = ratemaps.RateMaps(rm, arena_limits)
		# expected = np.array([1*0, 1*7., 1*3., 0*7., 0*3., 7.*3.])
		expected = np.array([1*0, 1*2., 1*7., 1*3., 1*4.,
							 0*2., 0*7, 0*3, 0*4,
							 2*7, 2*3, 2*4,
							 7*3, 7*4,
							 3*4])
		result = rmps.get_rate_products()
		np.testing.assert_array_almost_equal(expected, result, decimal=8)

	def test_get_distancehistogram_and_centers(self):
		rm = np.array([
			[1, 7.],
			[0, 3.],
		])
		arena_limits = np.array([[0, 1], [0, 1]])
		rmps = ratemaps.RateMaps(rm, arena_limits)
		# The only ocurring distances are 0.5 and sqrt_half
		# We use 2 bins
		n_expected = np.array([(0+7+0+21), (3+0)])
		n, centers = rmps.get_distancehistogram_and_centers(bins=2)
		np.testing.assert_array_almost_equal(n_expected, n, decimal=8)

