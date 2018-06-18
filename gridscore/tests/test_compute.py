__author__ = 'simonweber'
# from .. import compute
from gridscore import compute
import unittest
import numpy as np
import pandas as pd


class TestCompute(unittest.TestCase):
	def setUp(self):
		rawdata = {
			'spikepositions': np.array(
				[
					[0.4, 0.2],
					[0.1, 0.0],
					[-0.2, -0.9],
					[-1.0, 0.3],
				]
			),
			'spiketimes': np.array(
				[0.5, 1.5, 3.5, 3.6]
			),
			'arena_limits': np.array(
				[
					[-1.0, 1.0],
					[-1.0, 1.0]
				]
			)
		}
		self.cmpt = compute.Compute(params=None, rawdata=rawdata)

	def test_values_in_timewindow(self):
		window_start = 1.4
		window_end = 3.6
		expected = np.array([
			[0.1, 0.0],
			[-0.2, -0.9],
			[-1.0, 0.3],
		])
		values = self.cmpt.spikepositions
		result = self.cmpt.values_in_timewindow(
			 values, window_start, window_end)
		np.testing.assert_array_equal(expected, result)


	def test_get_timewindows(self):
		n = 4
		windowsize = 700.
		last_spiketime = 1600.
		### For sliding window of fixed size ###
		expected = np.array(
			[
				[0., 700.],
				[300., 1000.],
				[600., 1300.],
				[900., 1600.]
			]
		)
		result = self.cmpt.get_timewindows(n, last_spiketime, windowsize)
		np.testing.assert_array_equal(expected, result)
		### For windows that are decreased from the left ###
		last_spiketime = 1800.
		expected = np.array(
			[
				[0., last_spiketime],
				[600., last_spiketime],
				[1200., last_spiketime],
				[last_spiketime, last_spiketime]
			]
		)
		result = self.cmpt.get_timewindows(n, last_spiketime,
										   windowmethod='decrease_from_left')
		np.testing.assert_array_equal(expected, result)
		### For windows that are increased from zero ###
		n = 4
		last_spiketime = 1600.
		expected = np.array(
			[
				[0., 400.],
				[0., 800.],
				[0., 1200.],
				[0., last_spiketime]
			]
		)
		result = self.cmpt.get_timewindows(n, last_spiketime,
										   windowmethod='increase')
		np.testing.assert_array_equal(expected, result)
		### For windows that separate the time into n bins ###
		n = 4
		last_spiketime = 1600.
		expected = np.array(
			[
				[0, 400],
				[400, 800],
				[800, 1200],
				[1200, 1600],
			]
		)
		result = self.cmpt.get_timewindows(n, last_spiketime,
										   windowmethod='bins')
		np.testing.assert_array_equal(expected, result)


	def test_isgrid(self):
		symmetries = [2, 3, 4, 5, 6, 7, 8]
		gridscores_for_symmetries = [0.1, 0.3, -0.2, 0.6, 0.3, 0.61, 0.1]
		expected = False
		result = compute.Compute.isgrid(symmetries, gridscores_for_symmetries)
		self.assertEqual(expected, result)
		gridscores_for_symmetries = [0.1, 0.3, -0.2, 0.1, 0.62, 0.61, 0.1]
		expected = True
		result = compute.Compute.isgrid(symmetries, gridscores_for_symmetries)
		self.assertEqual(expected, result)

	# def test_baseline_and_trial_intervals_escobar(self):
	# 	# a = np.array(
	# 	# 	[
	# 	# 		('l1', 0, 10), ('d1', 10, 22)
	# 	# 	],
	# 	# 	 dtype=[
	# 	# 		 ('light', 'S10'), ('start', 'i4'), ('end', 'i4')
	# 	# 	 ])
	# 	# df = pd.DataFrame(a)
	# 	# print()
	# 	# np.savetxt('/Users/simonweber/programming/workspace/gridscore/tests'
	# 	# 		   '/test.txt', a)

	def test_baseline_and_trial_intervals_escobar(self):
		self.cmpt.params = {
			'dat': dict(filename='jp693-11062015-0108',
						publication='escobar2016')
		}
		baselines, trials = self.cmpt.baseline_and_trial_intervals_escobar()
		sampling_rate = self.cmpt.get_sampling_rate_escobar()
		expected_baselines = np.array(
			[
				[3., 0., 10443266 / sampling_rate],
				[2., 156849949 / sampling_rate, 168850451 / sampling_rate]
			]
		)
		np.testing.assert_array_almost_equal(expected_baselines, baselines)


