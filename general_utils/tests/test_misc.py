__author__ = 'simonweber'

import unittest
import numpy as np
from general_utils import misc


class TestMisc(unittest.TestCase):

	def test_get_hexagonal_coordinates(self):
		sidelength = 4
		expected = np.array(
			[
				[0, 0],

				[1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],

				[2, 0], [1, 1], [0, 2], [-1, 2], [-2, 2], [-2, 1], [-2, 0],
				[-1, -1], [0, -2], [1, -2], [2, -2], [2, -1],

				[3,0], [2, 1], [1, 2], [0, 3], [-1, 3], [-2, 3], [-3, 3],
				[-3, 2], [-3, 1], [-3, 0], [-2, -1], [-1, -2], [0, -3],
				[1, -3], [2, -3], [3, -3], [3, -2], [3, -1],


			]
		)
		expected = np.sort(expected, axis=0)
		result = np.sort(misc.get_hexagonal_coordinates(sidelength), axis=0)
		np.testing.assert_array_equal(result, expected)

	def test_hex2cart(self):
		# The grid spacing
		size = 1.
		# Test for the central locations and the six surrounding locations
		# Each 4-tupel is:
		# (column_hex_coord, row_hex_coord, expected_x, expected_y)
		# Note that hexagonal coordinates are represented in axial system
		# y = size * np.tan(np.pi/6) / 2
		y = size * 3 / (2 * np.sqrt(3))
		tpls = [
			(0, 0, 0, 0),
			(1, 0, size, 0),
			(0, 1, size/2, -y),
			(-1, 1, -size/2, -y),
			(-1, 0, -size, 0),
			(0, -1, -size/2, y),
			(1, -1, size/2, y)
		]

		for q, r, expected_x, expected_y in tpls:
			print((q, r, expected_x, expected_y))
			result_x, result_y = misc.hex2cart(q, r, size=size)
			self.assertEqual(expected_x, result_x)
			self.assertEqual(expected_y, result_y)

	def test_idx2loc(self):
		radius = 0.5
		nbins = 3
		idx = 0
		expected = - radius
		result = misc.idx2loc(idx, radius, nbins)
		self.assertEqual(result, expected)
		idx = nbins - 1
		expected = radius
		result = misc.idx2loc(idx, radius, nbins)
		self.assertEqual(result, expected)
		idx = 1.01
		expected = 0.0
		result = misc.idx2loc(idx, radius, nbins)
		self.assertEqual(result, expected)
