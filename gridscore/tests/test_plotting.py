__author__ = 'simonweber'
import unittest
import numpy as np
from .. import plotting

class TestPlot(unittest.TestCase):
    def setUp(self):
        self.spikepositions = np.array([
            [0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]
        ])
        self.arena_limits = np.array([[0, 1], [0, 1]])
        self.plot = plotting.Plot(spikepositions=self.spikepositions,
                             arena_limits=self.arena_limits)

    def test_set_nonspecified_arguments_to_default(self):
        # self.shell_limits_choice = 'automatic_double'
        # self.cut_off_position = 0.5
        arg_default_dict = dict(
            a='foo', b=0.5, c=1e3, d='bar'
        )
        self.plot.a = 'hello'
        self.plot.c = 'one'
        self.plot.set_nonspecified_attributes_to_default(arg_default_dict)
        self.assertEqual(self.plot.a, 'hello')
        self.assertEqual(self.plot.b, 0.5)
        self.assertEqual(self.plot.c, 'one')
        self.assertEqual(self.plot.d, 'bar')

    def test_spike_idx(self):
        plot = self.plot
        spiketimes = np.array([0.1, 0.4, 1.5, 2.8, 2.9])
        plot_with_spiketimes = plotting.Plot(spikepositions=self.spikepositions,
                             arena_limits=self.arena_limits,
                            spiketimes=spiketimes)
        # If None
        n = None
        expected = np.array([0, 1, 2, 3, 4])
        result = plot.spike_idx(n)
        np.testing.assert_array_equal(expected, result)
        # If an integer is given
        n = 3
        expected = np.array([0, 1, 2])
        result = plot.spike_idx(n)
        np.testing.assert_array_equal(expected, result)
        # If an index array is given
        n = np.array([1, 3])
        expected = n
        result = plot.spike_idx(n)
        np.testing.assert_array_equal(expected, result)
        # If a single time interval is given
        n = np.array([[0.4, 2.85]])
        expected = np.array([1, 2, 3])
        result = plot_with_spiketimes.spike_idx(n)
        np.testing.assert_array_equal(expected, result)
        # If several time intervals are given
        n = np.array([[0.2, 0.3], [1.4, 2.85]])
        expected = np.array([2, 3])
        result = plot_with_spiketimes.spike_idx(n)
        np.testing.assert_array_equal(expected, result)
        # If intervals are given, but spiketimes are None.
        n = np.array([[0.2, 0.3], [1.4, 2.85]])
        self.assertRaises(ValueError, plot.spike_idx, n)

    def test_time_intervals_escobar(self):
        self.plot.computed = {}
        self.plot.computed['baselines'] = np.array(
            [
                [3.0, 0.0, 59.1],
                [2.0, 120.1, 200.0]
            ]
        )
        self.plot.computed['trials'] = np.array(
            [
                [2.0, 70., 80.],
                [-2.0, 80., 90.],
                [3.0, 90., 100.],
                [-3.0, 100., 110.],
                [2.0, 110., 120.],
                [-2.0, 120., 130.],
            ]
        )
        # Baseline start
        expected = np.array([[0.0, 59.1]])
        result = self.plot.get_time_intervals_of_trial_type(trial='baseline_start')
        np.testing.assert_array_equal(expected, result)
        # Light 2: l2
        expected = np.array([
            [70., 80.], [110., 120.]
        ])
        result = self.plot.get_time_intervals_of_trial_type(trial='l2')
        np.testing.assert_array_equal(expected, result)

    def test_concatenate_time_intervals(self):
        self.plot.computed = {}
        self.plot.computed['baselines'] = np.array(
            [
                [3.0, 0.0, 59.1],
                [2.0, 120.1, 200.0]
            ]
        )
        self.plot.computed['trials'] = np.array(
            [
                [2.0, 70., 80.],
                [-2.0, 80., 90.],
                [3.0, 90., 100.],
                [-3.0, 100., 110.],
                [2.0, 110., 120.],
                [-2.0, 120., 130.],
            ]
        )
        # Test for a single trial specification: l2 only
        trials = ['l2']
        expected = np.array(
            [
                [70., 80.], [110., 120.]
            ]
        )
        result = self.plot.concatenate_time_intervals(trials=trials)
        np.testing.assert_array_equal(expected, result)
        # Concatenate l2 and d2
        # NB: This leads to first all l2 and then all d2 time windows.
        trials = ['l2', 'd2']
        expected = np.array(
            [
                [70., 80.], [110., 120.], [80., 90.], [120., 130.]
            ]
        )
        result = self.plot.concatenate_time_intervals(trials=trials)
        np.testing.assert_array_equal(expected, result)

    def test_shift_timewindows_to_relative_timewindows(self):
        timewindows = np.array(
            [
                [0., 1.], [1., 2.], [1.5, 2.5], [4., 6.]
            ]
        )
        expected = np.array(
            [
                [0., 1.], [0., 1.], [0., 1.], [0., 2.]
            ]
        )
        result = self.plot.shift_timewindows_to_relative_timewindows(
            timewindows)
        np.testing.assert_array_equal(expected, result)

    def test_crop_intervals(self):
        intervals = np.array([
            [0., 100.], [100., 200.], [300., 500.]
        ])
        # First 10
        start = 0
        length = 10
        expected = np.array([
            [0., 10.], [100., 110.], [300., 310.]
        ])
        result = self.plot.crop_intervals(intervals, start, length)
        np.testing.assert_array_equal(expected, result)
        # 10 to 20
        start = 10
        length = 10
        expected = np.array([
            [10., 20.], [110., 120.], [310., 320.]
        ])
        result = self.plot.crop_intervals(intervals, start, length)
        np.testing.assert_array_equal(expected, result)

    def test_get_time_blocks(self):
        resolution = 30
        total_trial_length = 120
        expected = np.array([
            [0, 30], [30, 60], [60, 90], [90, 120]
        ])
        result = self.plot.get_time_blocks(resolution, total_trial_length)
        np.testing.assert_array_equal(expected, result)