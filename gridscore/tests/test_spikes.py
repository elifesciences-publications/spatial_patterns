__author__ = 'simonweber'
import unittest
import numpy as np
from .. import spikes
from general_utils.misc import pol2cart
from general_utils.misc import cart2pol
# from gridscore.spikes import spikes

class TestSpikes(unittest.TestCase):
	def setUp(self):
		angle0 = 0
		angle1 = np.pi / 3
		angle2 = 2 * angle1
		angle3 = 3 * angle1
		angle4 = 4 * angle1
		angle5 = 5 * angle1
		rho = 1
		positions = np.array(
			[
				[0, 0],
				pol2cart(rho, angle0),
				pol2cart(rho, angle1),
				pol2cart(rho, angle2),
				pol2cart(rho, angle3),
				pol2cart(rho, angle4),
				pol2cart(rho, angle5),
			]
		)
		arena_limits = np.array([[-1, 1], [-1, 1]])
		self.spikes = spikes.Spikes(positions,
									arena_limits=arena_limits)

		ratemap = np.array(
			[
				[2., 5.],
				[3., 0.]
			]
		)
		self.ratemap = spikes.SpikesFromRatemap(
			ratemap, arena_limits=arena_limits
		)

	def test_neighbor_indeces(self):
		"""
		Test if all neighbor indeces are found correctly
		for an infinite shell
		"""
		idx = 3
		expected = np.array([0, 1, 2, 4, 5, 6])
		self.spikes.shell_limits = None
		result = self.spikes._neighbor_indeces(idx)
		np.testing.assert_array_equal(expected, result)

	def test_neighbor_positions_all(self):
		self.spikes.neighborhood = 'all'
		idx = 3
		expected = np.array(
			[
				self.spikes.pos[0],
				self.spikes.pos[1],
				self.spikes.pos[2],
				self.spikes.pos[4],
				self.spikes.pos[5],
				self.spikes.pos[6],
			]
		)
		self.spikes.shell_limits = None
		result = self.spikes._neighbor_positions(idx)
		np.testing.assert_array_equal(expected, result)

	def test_neighbor_positions_shell(self):
		# Single shell
		self.spikes.shell_limits = np.asarray([0.7, 1.3])
		idx = 3
		expected = np.array([2, 0, 4]).sort()
		result = self.spikes._neighbor_indeces(idx).sort()
		np.testing.assert_array_equal(expected, result)
		# Double shell
		self.spikes.shell_limits = np.asarray([[0.7, 1.3], [1.9, 2.1]])
		idx = 3
		expected = np.array([2, 0, 4, 6]).sort()
		result = self.spikes._neighbor_indeces(idx).sort()
		np.testing.assert_array_equal(expected, result)

	def test_psi_n(self):
		# Central spike
		idx = 0
		expected = 1
		self.spikes.n_symmetry = 6
		self.spikes.shell_limits = None
		result = self.spikes._psi_n(idx)
		self.assertAlmostEqual(expected, result, 8)
		# Decentral spike
		idx = 1
		angle = cart2pol(1.5, np.sqrt(3. / 4))[1]
		expected = np.abs(
			2 * np.exp(1j * 6 * 0)
			+ 2 * np.exp(1j * 6 * np.pi / 3)
			+ 2 * np.exp(1j * 6 * angle)
		) / (self.spikes.pos.shape[0] - 1)
		result = np.abs(self.spikes._psi_n(idx))
		print(result)
		self.assertAlmostEqual(expected, result, 8)

	def test_get_index_distribution(self):
		k = 1
		# Probability mass function at index 1
		expected = 0.5
		result = self.ratemap.get_index_distributions().pmf(k=k)
		self.assertAlmostEqual(expected, result, 8)

	def test_get_spiketimes_spikenumbers(self):
		self.spikes.times = np.array([3.2, 4., 7., 9., 12., 16., 18., 19.])
		# Without leading 0 and every 3rd element
		every_nth = 3
		spiketimes_expected = np.array([7., 16.])
		spikenumbers_expected = np.array([3, 6])
		spiketimes, spikenumbers = self.spikes.get_spiketimes_spikenumbers(
			every_nth=every_nth, include_zero=False)
		np.testing.assert_array_equal(spiketimes, spiketimes_expected)
		np.testing.assert_array_equal(spikenumbers, spikenumbers_expected)
		# If it should include the zero
		every_nth = 2
		spiketimes_expected = np.array([0, 4., 9., 16., 19.])
		spikenumbers_expected = np.array([0, 2, 4, 6, 8])
		spiketimes, spikenumbers = self.spikes.get_spiketimes_spikenumbers(
			every_nth=every_nth, include_zero=True)
		np.testing.assert_array_equal(spiketimes, spiketimes_expected)
		np.testing.assert_array_equal(spikenumbers, spikenumbers_expected)