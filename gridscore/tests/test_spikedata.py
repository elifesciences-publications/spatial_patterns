__author__ = 'simonweber'
import unittest
import numpy as np
from .. import spikedata

class TestSpikeData(unittest.TestCase):
	def setUp(self):
		self.sd_escobar = spikedata.SpikeData(publication='escobar2016',
								 filename='jp693-11062015-0108',
											  identifier=13)
		self.sd_escobar.time_between_positions = (
				self.sd_escobar.get_samples_between_position_samples_escobar() /
				self.sd_escobar.get_sampling_rate_escobar()
			)

	# Escobar 2016
	def test_get_spikepositions_escobar(self):
		# The 17th spike of cluster 13 is in .clu in line 2728 so entry 2727
		# In .res, line 2727 is 150378
		# 150378 / sampling_rate = 7.5189 seconds
		# 7.5189 seconds is closest to 7.52 seconds
		# 7.52 / 0.02 = 376
		expected = self.sd_escobar.ratpositions[376]
		# Fourth spike
		spike_number = 16 # 17 - 1
		spikepositions = self.sd_escobar.get_spikepositions()
		result = spikepositions[spike_number]
		np.testing.assert_array_almost_equal(expected, result)

	def test_get_positiontimes_escobar(self):
		positiontimes = self.sd_escobar.get_positiontimes_escobar(
			self.sd_escobar.time_between_positions)
		# Look at position 300
		n = 299
		expected = self.sd_escobar.time_between_positions  * n
		result = positiontimes[n]
		np.testing.assert_array_almost_equal(expected, result, decimal=10)

	def test_get_sampling_rate_escobar(self):
		expected = 2e4
		result = self.sd_escobar.get_sampling_rate_escobar()
		self.assertEqual(expected, result)

	def test_get_samples_between_position_samples_escobar(self):
		expected = 400
		result = self.sd_escobar.get_samples_between_position_samples_escobar()
		self.assertEqual(expected, result)

	def test_get_ratpositions_escobar(self):
		ratpositions = self.sd_escobar.get_ratpositions_escobar()
		# Look at position 300
		n = 299
		expected = np.array([633.398, 474.008])
		result = ratpositions[n]
		np.testing.assert_array_almost_equal(expected, result, decimal=10)

	def test_get_spiketimes_escobar(self):
		# With identifier 13 the first spike is in line 40 in .clu. This
		# corresponds to entry 39 so in Numpy indexing: index 38
		# In .res line 39 has the value 1707.
		# In seconds this is: 1707 / sampling_rate
		# With sampling_rate = 2e4, this gives: 0.08535 seconds
		expected = 0.08535
		result = self.sd_escobar.get_spiketimes_escobar()[0]
		self.assertAlmostEqual(expected, result, places=5)

	# Sargolini 2006
	def test_get_ratpositions_positiontimes_spiketimes_sargolini(self):
		sd = spikedata.SpikeData(publication='sargolini2006',
								 filename='10073-17010302_t1c1')
		ratpositions, \
		positiontimes, \
		spiketimes, \
		ratpositions2, \
		headdirections = sd.get_ratpos_post_spiket_ratpos2_hds_sargolini()
		# Look at position 10 and the 10th spike
		n = 9
		result = [ratpositions[n, 0],
				  ratpositions[n, 1],
				  positiontimes[n],
				  spiketimes[n],
				  ]
		expected = [0.27392229641919036,
					17.491088341792903,
					0.17999999999994998,
					2.7457708333333333,
					]
		for r, e in zip(result, expected):
			self.assertAlmostEqual(r, e, places=10)

	def test_get_ratpositions_positiontimes_spiketimes_hafting(self):
		sd = spikedata.SpikeData(publication='hafting2005',
								 filename='figure2d_trial1',
								 identifier='rat11015_t1c1')
		ratpositions, \
		positiontimes, \
		spiketimes, \
		ratpositions2, \
		headdirections = sd.get_ratpositions_positiontimes_spiketimes_hafting()
		# Look at position 10 and the 10th spike
		n = 9
		result = [ratpositions[n, 0],
				  ratpositions[n, 1],
				  positiontimes[n],
				  spiketimes[n],
				  ]
		expected = [54.81724236899061,
					-28.63819446133517,
					0.1799999999998363,
					10.131822916666666,
					]
		for r, e in zip(result, expected):
			self.assertAlmostEqual(r, e, places=10)

	def test_get_spikepositions(self):
		sd = spikedata.SpikeData(publication='sargolini2006',
								 filename='10073-17010302_t1c1')
		sd.ratpositions = np.array([[0, 1], [2, 3], [4, 5]])
		sd.positiontimes = np.array([0, 1, 2])
		sd.spiketimes = np.array([1.1, 1.8])
		expected = np.array([[2, 3], [4, 5]])
		result = sd.get_spikepositions()
		np.testing.assert_array_equal(expected, result)
