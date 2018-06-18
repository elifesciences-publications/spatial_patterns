__author__ = 'simonweber'

import unittest
import numpy as np
from general_utils import arrays
import matplotlib.pyplot as plt

class TestArrays(unittest.TestCase):

	def test_nonzero_and_zero_elements_to_constant(self):
		a =  np.array([0., 0.2, 100.2, -5.1, 0, 1])
		c_zero = 0.5
		c_nonzero = 7.
		expected = np.array([c_zero, c_nonzero, c_nonzero, c_nonzero,
							 c_zero, c_nonzero])
		result = arrays.nonzero_and_zero_elements_to_constant(
			a, c_zero=c_zero, c_nonzero=c_nonzero)
		np.testing.assert_array_equal(result, expected)

	def test_binedges2centers(self):
		# Odd number of edges
		binedges = np.array([0., 0.5, 1.])
		expected = np.array([0.25, 0.75])
		result = arrays.binedges2centers(binedges)
		np.testing.assert_array_equal(result, expected)
		# Even number of edges
		binedges = np.array([0., 0.5, 1., 1.5])
		expected = np.array([0.25, 0.75, 1.25])
		result = arrays.binedges2centers(binedges)
		np.testing.assert_array_equal(result, expected)

	def test_shift_positions(self):
		positions = np.array(
			[
				[-0.2, 0.4],
				[0.0, -0.3],
				[0.1, -0.3]
			]
		)
		shift = np.array([0.1, -0.8])
		expected = np.array(
			[
				[-0.1, -0.4],
				[0.1, -1.1],
				[0.2, -1.1]
			]
		)
		result = arrays.shift_positions(positions, shift)
		np.testing.assert_array_equal(result, expected)

	def test_rotate_positions(self):
		positions = np.array(
			[
				[1.0, 0.0],
				[0.2, 0.2],
				[0.0, -0.5]
			]
		)
		theta = 90
		expected = np.array(
			[
				[0.0, 1.0],
				[-0.2, 0.2],
				[0.5, 0.0]
			]
		)
		result = arrays.rotate_positions(positions, theta)
		np.testing.assert_almost_equal(result, expected, decimal=10)

	def test_circular_center_of_mass(self):
		a = [0, 0, 0, 10, 0]
		angles = [0, np.pi/4., np.pi/2., 3/4 * np.pi, np.pi]
		expected = 3/4 * np.pi
		result = arrays.circular_center_of_mass(a, angles)
		self.assertEqual(result, expected)

	def test_find_nearest(self):
		# One dimension
		a = np.array([0.2, 0.3, 0.11, 0.0])
		v = 0.1
		expected_element = 0.11
		result_element = arrays.find_nearest(a, v, ret='element')
		self.assertEqual(result_element, expected_element)
		expected_index = 2
		result_index = arrays.find_nearest(a, v, ret='index')
		self.assertEqual(result_index, expected_index)
		# Higher dimensions
		a = np.array(
			[
				[0.2, 0.12],
				[0.13, 0.3],
				[0.14, 0.4]
			]
		)
		# For axis = None
		expected_element = 0.12
		result_element = arrays.find_nearest(a, v, ret='element')
		self.assertEqual(result_element, expected_element)
		# For axis 0
		expected_elements = [0.13, 0.12]
		result_elements = arrays.find_nearest(a, v, ret='element', axis=0)
		np.testing.assert_array_equal(result_elements, expected_elements)
		# For axis 1
		expected_elements = [0.12, 0.13, 0.14]
		result_elements = arrays.find_nearest(a, v, ret='element', axis=1)
		np.testing.assert_array_equal(result_elements, expected_elements)

	def test_remove_nan_rows(self):
		a = np.array(
			[
				[1, 2],
				[np.nan, 3],
				[np.nan, np.nan],
				[4, 5],
				[6, np.nan],
				[7, 8]
			]
		)
		expected = np.array(
			[
				[1, 2],
				[4, 5],
				[7, 8]
			]
		)
		result = arrays.remove_nan_rows(a,)
		np.testing.assert_array_equal(result, expected)


	def test_positions_within_circle(self):
		positions = np.array([
			[-0.3, -0.3],
			[-0.2,  0.3],
			[0.3, -0.05],
			[0.0, 0.0],
			[0.4, -0.4],
			[0.4, 0.4]
		])
		origin = [-0.2, 0.1]
		radius = 0.25
		# Return positions
		expected = [[-0.2, 0.3], [0.0, 0.0]]
		result = arrays.positions_within_circle(positions, origin, radius,
												ret='positions')
		np.testing.assert_array_equal(expected, result)
		# Return index
		expected = [1, 3]
		result = arrays.positions_within_circle(positions, origin, radius,
												ret='index')
		np.testing.assert_array_equal(expected, result)
		# # Plot
		# plt.scatter(positions[:, 0], positions[:, 1])
		# circle = plt.Circle(origin,
		# 		   radius, fc='none', lw=2,
		# 		   linestyle='dashed')
		# plt.gca().add_artist(circle)
		# plt.show()

	def test_concatenated_array_from_list_of_arrays(self):
		l = [np.array([1]), np.array([1., 2.]), np.array([0.5, 0.6, 0.7])]
		expected = np.array([1., 1., 2., 0.5, 0.6, 0.7])
		result = arrays.concatenated_array_from_list_of_arrays(l)
		np.testing.assert_array_equal(expected, result)