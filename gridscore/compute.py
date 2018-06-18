import numpy as np
from .plotting import Plot
import general_utils.misc
from .spikes import Spikes

from .spikedata import MAINDIR
from .spikedata import PATHS_DICTIONARY
from .spikedata import SpikeData
import os
import pandas as pd
from .dictionaries import str_to_int_escobar
from .dictionaries import STD_THRESHOLD

class Compute(Plot, SpikeData):
    """
    Class to compute quantities from the rawdata

    Parameters
    ----------
    params : dict
    rawdata : dict
    """

    def __init__(self, params, rawdata, computed=None):
        self.params = params
        self.rawdata = rawdata
        self.computed = computed
        # Frequently used rawdata get its own variable for later convenience
        self.spikepositions = rawdata['spikepositions']
        self.arena_limits = rawdata['arena_limits']
        self.spiketimes = rawdata['spiketimes']

    def isgrid_time_evo(self,
                           method='automatic_single',
                           windowmethod='increase',
                           windowsize=None,
                           std_threshold=STD_THRESHOLD,
                           gridscore_norm=None, from_final=False,
                           sigma=5, drop_recalc=0,
                           n_windows=201):

        std_threshold = std_threshold
        timewindows = self.get_timewindows(
            n=n_windows, last_spiketime=self.spiketimes[-1],
            windowsize=windowsize,
            windowmethod=windowmethod,
        )
        # gridscores_for_nsymmetry.
        # Final shape: (number of symmetries, number of timewindows)
        gridscores_for_symmetries_for_all_times = []
        symmetries = np.array([2, 3, 4, 5, 6, 7])
        for n_symmetry in symmetries:
            gridscores = self._gridscores_in_timewindows(
                timewindows, method=method,
                gridscore_norm=gridscore_norm,
                std_threshold=std_threshold,
                windowmethod=windowmethod,
                sigma=sigma,
                drop_recalc=drop_recalc,
                n_symmetry=n_symmetry
            )[:, 1] # The second column is the gridscore
            gridscores_for_symmetries_for_all_times.append(gridscores)

        gridscores_for_symmetries_for_all_times = np.asarray(
            gridscores_for_symmetries_for_all_times)
        isgrid_for_all_times = []
        for t in np.arange(n_windows):
            isgrid_for_all_times.append(
                self.isgrid(symmetries,
                            gridscores_for_symmetries_for_all_times[:, t])
            )
        return np.array(isgrid_for_all_times)

    @staticmethod
    def isgrid(symmetries, gridscores_for_symmetries):
        symmetries = np.asarray(symmetries)
        gridscores_for_symmetries = np.asarray(gridscores_for_symmetries)
        return (np.argmax(gridscores_for_symmetries)
                == np.argwhere(symmetries == 6)[0, 0])

    # def gridscore_time_evo(self,
    #                         method='automatic_single', windowmethod='increase',
    #                         windowsize=None,
    #                         std_threshold=STD_THRESHOLD,
    #                         gridscore_norm=None, from_final=False,
    #                         sigma=5, drop_recalc=0,
    #                         n_windows=201,
    #                         compare_to_other_symmetries=False,
    #                         mode=None):
    #     """
    #     Computes the gridscore at each spike time
    #
    #     Parameters
    #     ----------
    #     mode : str
    #         'automatic_single_final', the grid score is obtained from the
    #         psi_6 values of each spike at the final time by taking the
    #         mean of the psi_6 values of all the spikes that have occured
    #         so far. This might actully indicate if early spikes are already
    #         part of the final grid. That is the case, if early spikes have
    #         a high final psi_6 value.
    #
    #         'automatic_single', the grid score is from all the spike locations
    #         that have occurred so far. In contrast to 'automatic_single_final'
    #         the psi_6 values are calculated only from that spike locations
    #         that occurred to the current moment in time, and not at the end.
    #         This calculation takes much longer, because the histogram and the
    #         psi_6 values need to be calculated at each time step. This
    #         measure will fluctuate at early times, when only few spike locations
    #         are present.
    #
    #         'sargolini', the grid score is obtained from the correlogram of
    #         all spike locations that have occurred up to this point in time.
    #         This computation takes a long time.
    #     """
    #     std_threshold = std_threshold
    #     # Check if timewindows are part of the way the gridscore is computed
    #     # and obtain them if necessary
    #     if self._timewindows_needed(mode):
    #         timewindows = self.get_timewindows(
    #             n=n_windows, last_spiketime=self.spiketimes[-1],
    #             windowsize=windowsize,
    #             windowmethod=windowmethod)
    #     if from_final:
    #         n = self.time_intervals_from_mode(mode)
    #         idx = self.spike_idx(n=n)
    #         spikes = self.get_spikes_class_for_spike_selection(idx=idx)
    #         shell_limits = spikes.get_shell_limits(
    #                                 shell_limits_choice=method,
    #                                 cut_off_position='arena_fraction',
    #                                 maximum_index=0)
    #         print(shell_limits)
    #         psi_n = spikes.psi_n_all(shell_limits=shell_limits,
    #                     normalization=gridscore_norm,
    #                     std_threshold=std_threshold,
    #                     compare_to_other_symmetries=compare_to_other_symmetries)
    #         if compare_to_other_symmetries == 4:
    #             psi_real = np.real(psi_n)
    #             psi_abs = psi_real - np.amin(psi_real)
    #         else:
    #             psi_abs = np.absolute(psi_n)
    #         if 'each_spike_final' in mode:
    #             print('each_spike_final')
    #             gridscores = psi_abs
    #         else:
    #             # norm = np.arange(len(psi_abs)) + 1
    #             # gridscores = np.cumsum(psi_abs) / norm
    #             gridscores = self.statistics_within_timewindows(psi_abs,
    #                                                             timewindows,
    #                                                             windowmethod)
    #     else:
    #         gridscores = self._gridscores_in_timewindows(
    #             timewindows, method=method,
    #             gridscore_norm=gridscore_norm,
    #             std_threshold=std_threshold,
    #             windowmethod=windowmethod,
    #             sigma=sigma,
    #             drop_recalc=drop_recalc,
    #             compare_to_other_symmetries=compare_to_other_symmetries
    #         )
    #
    #     return gridscores

    def _get_idx_spikes_cls_and_shell_limits(self, n, method):
        """Convenience function"""
        idx = self.spike_idx(n=n)
        spikes =  self.get_spikes_class_for_spike_selection(idx=idx)
        shell_limits = spikes.get_shell_limits(
                                shell_limits_choice=method,
                                cut_off_position='arena_fraction',
                                maximum_index=0)
        return idx, spikes, shell_limits

    def gridscore_for_each_spike(self,
                            method='automatic_single',
                            std_threshold=STD_THRESHOLD,
                            gridscore_norm=None,
                            compare_to_other_symmetries=False,
                            mode=None, **kwargs):
        """
        Computes gridscore for a selection of spikes.
        NB: Kind of unnecessary, because we can do the same with
        `gridscore_for_each_spike_in_reference_to_subset`. But this function
        here is faster.
        """
        t_intervals = self.get_time_intervals_from_mode(mode)
        idx, spikes, shell_limits = self._get_idx_spikes_cls_and_shell_limits(
            t_intervals, method)

        psi_n = spikes.psi_n_all(shell_limits=shell_limits,
                normalization=gridscore_norm,
                std_threshold=std_threshold,
                compare_to_other_symmetries=compare_to_other_symmetries)

        gridscores = self._get_psi_abs(psi_n, compare_to_other_symmetries)
        return gridscores

    def gridscore_for_each_spike_in_reference_to_subset(self,
                            method='automatic_single',
                            compare_to_other_symmetries=True,
                            mode=None, **kwargs):
        """
        Computes grid score of each spike in reference to subset of spikes
        """
        # Get the time intervals in which the reference spikes occur
        t_intervals_reference = self.get_time_intervals_from_mode(mode)
        # Get index into the references spikes, the spikes class and the
        # shell limits computed for the reference spikes
        idx, spikes, shell_limits_reference = \
            self._get_idx_spikes_cls_and_shell_limits(
                t_intervals_reference, method)
        # Get index into all spikes
        idx_all = self.spike_idx(n=None)
        # The list of psi values for every spike in reference to all the
        # spike in the reference set.
        psis = []
        # Get the index into the added spike (since it starts at 0, this is
        # the len(idx) and not len(idx) + 1
        idx_last = len(idx)
        for i in idx_all:
            # Create a new spikes class, of all spikes in the reference set
            # and the spike that is currently under investigation
            spikes_one_added = self.get_spikes_class_for_spike_selection(
                idx=np.concatenate((idx, np.array([i]))
                )
            )
            # Set the shell limits of this class to the reference shell
            # limits, just to be sure that the new spike does not mess up the
            # determination of the shell limits (unlikely, but well ...).
            spikes_one_added.shell_limits = shell_limits_reference
            psis.append(
                spikes_one_added._psi_n(
                    idx_last,
                    compare_to_other_symmetries=compare_to_other_symmetries)
            )
        gridscores = self._get_psi_abs(psis, compare_to_other_symmetries)
        return gridscores

    def gridscore_time_evo_in_windows(self,
                            method='automatic_single', windowmethod='increase',
                            windowsize=None,
                            std_threshold=STD_THRESHOLD,
                            gridscore_norm=None, from_final=False,
                            sigma=5, drop_recalc=0,
                            n_windows=201,
                            compare_to_other_symmetries=False):
        """
        The time evolution of gridscores computed in time windows.
        """
        std_threshold = std_threshold
        timewindows = self.get_timewindows(
            n=n_windows, last_spiketime=self.spiketimes[-1],
            windowsize=windowsize,
            windowmethod=windowmethod)
        if from_final:
            idx, spikes, shell_limits = \
                self._get_idx_spikes_cls_and_shell_limits(None, method)
            print(shell_limits)
            psi_n = spikes.psi_n_all(shell_limits=shell_limits,
                        normalization=gridscore_norm,
                        std_threshold=std_threshold,
                        compare_to_other_symmetries=compare_to_other_symmetries)
            psi_abs = self._get_psi_abs(psi_n, compare_to_other_symmetries)
            gridscores = self.statistics_within_timewindows(psi_abs,
                                                            timewindows,
                                                            windowmethod)
        else:
            gridscores = self._gridscores_in_timewindows(
                timewindows, method=method,
                gridscore_norm=gridscore_norm,
                std_threshold=std_threshold,
                windowmethod=windowmethod,
                sigma=sigma,
                drop_recalc=drop_recalc,
                compare_to_other_symmetries=compare_to_other_symmetries
            )

        return gridscores

    @staticmethod
    def _get_psi_abs(psi_n, compare_to_other_symmetries):
        if compare_to_other_symmetries == 4:
            psi_real = np.real(psi_n)
            psi_abs = psi_real - np.amin(psi_real)
        else:
            psi_abs = np.absolute(psi_n)
        return psi_abs

    @staticmethod
    def _timewindows_needed(mode):
        """Convenience function to check if timewindows are needed"""
        if not 'each_spike_final' in mode:
            tw_needed = True
        else:
            tw_needed = False
        return tw_needed

    @staticmethod
    def isfinal(d):
        """
        Check if d has key 'from_final' with boolian value and returns
        True if it has and it is True and False otherwise.
        """
        try:
            return d['from_final']
        except KeyError as e:
            print(e)
            print('WARNING:Could not be determined if grid'
                  'score should be computed'
                  'from final spike distribution. It is assumed that not.')
            return False

    # @staticmethod
    # def iswindow(d):
    # 	"""
    # 	Check if d has key 'window'
    # 	"""
    # 	try:
    # 		d['window']
    # 		return True
    # 	except KeyError as e:
    # 		print(e)
    # 		print('WARNING:Could not be determined if grid'
    # 			  'score should be computed'
    # 			  'from final spike distribution. It is assumed that not.')
    # 		return False


    def _get_gridscore_at_time_steps(self, method, stepsize=1,
                                     gridscore_norm=None, std_threshold=None):
        """
        See gridscore_time_evolution for documentation

        Parameters
        ----------
        stepsize : int
            Ever stepsize-th step is taken. For many spike times it might
            be useful to skip some, to save computation time.
        """
        gridscores = []
        steps = np.arange(1, len(self.spikepositions) + 1, stepsize)
        for step in steps:
            print(step)
            spikepos = self.spikepositions[:step]
            spikes = Spikes(positions=spikepos,
                            arena_limits=self.arena_limits)
            gridscore = spikes.get_gridscore(method=method,
                                             gridscore_norm=gridscore_norm,
                                             std_threshold=std_threshold)
            gridscores.append(gridscore)
        gridscores = np.array(gridscores)
        return gridscores

    def values_in_timewindow(self, values, window_start, window_end):
        """
        Returns values whose spike times are within time window

        Note that the window is inclusive, both in the beginning and
        at the end.

        Parameters
        ----------
        values : ndarray of shape (N, ...), where N = len(self.spiketimes)
            For example:
                spikepositions: shape (N, 2)
                gridscores from final configuration: shape (N)
        window_start : float
            Specifies the beginning of the time window.
        window_end : float
            Analogous to `window_start`

        Returns
        -------
        spikepositions : ndarray of shape (N, 2)
        """
        condition = np.logical_and(window_start <= self.spiketimes,
                                   self.spiketimes <= window_end)
        return values[condition]

    def _gridscores_in_timewindows(self, timewindows,
                                      method,
                                      gridscore_norm=None,
                                      std_threshold=None,
                                      windowmethod=None,
                                      sigma=5,
                                      drop_recalc=0,
                                      n_symmetry=6,
                                      compare_to_other_symmetries=False):
                                      # windowmethod='fixed_size'):
        """
        Gridscores for only the spikes that occurred in the given timewindows

        Takes only those spikes that occurred in the given timewindows and
        computes their grid scores from the resulting spike maps.

        NB: This is different from get_gridscores_in_timewindows in
        plotting.py, where the grid scores are computed for many spike
        positions and then the grid scores for spikes that occurred in the
        given timewindows is returned.

        Parameters
        ----------
        timewindows : ndarray of shape (n, 2)
            See get_timewindows
        method : str {'automatic_single', 'sargolini', 'langston', ...}
            The grid score method
        gridscore_norm : str or None
        std_threshold : str or None
        windowmethod : str
            This is only needed here, to select the times correctly

        Returns
        -------
        t_gs_gsstd : ndarray of shape (n, 3)
            times, gridscores and gridscore standard deviations within
            each time window.
        """
        t_gs_gsstd = []
        for tw in timewindows:
            print(tw)
            spikepos = self.values_in_timewindow(
                    self.spikepositions, tw[0], tw[1])
            spikes = Spikes(positions=spikepos,
                            arena_limits=self.arena_limits)
            gridscore, gridscore_std = spikes.get_gridscore(method=method,
                                             gridscore_norm=gridscore_norm,
                                             std_threshold=std_threshold,
                                             return_std=True,
                                             sigma=sigma,
                                            drop_recalc=drop_recalc,
                                            n_symmetry=n_symmetry,
                    compare_to_other_symmetries=compare_to_other_symmetries)
            time = self.get_time_from_timewindow(tw, windowmethod)
            t_gs_gsstd.append([time, gridscore, gridscore_std])
        t_gs_gsstd = np.array(t_gs_gsstd)
        return t_gs_gsstd

    def statistics_within_timewindows(self, a, timewindows, windowmethod):
        t_mean_std = []
        for tw in timewindows:
            print(tw)
            a_in_window = self.values_in_timewindow(a, tw[0], tw[1])
            mean = np.nanmean(a_in_window)
            std = np.nanstd(a_in_window)
            time = self.get_time_from_timewindow(tw, windowmethod)
            t_mean_std.append([time, mean, std])
        t_mean_std = np.array(t_mean_std)
        return t_mean_std

    @staticmethod
    def get_time_from_timewindow(timewindow, windowmethod):
        if (windowmethod == 'fixed_size'
            or windowmethod == 'decrease_from_left'
            or windowmethod == 'bins'):
            time = timewindow[0]
        else:
            time = timewindow[1]
        return time

    def baseline_and_trial_intervals_escobar(self):
        """
        Returns the time intervals for baseline and trials

        Two arrays, the baseline array  of shape (2, 3)
        the trials array of shape (N, 3)
        The three colums:
        * First column denotes part of the trial:
        Denoting convention:
        1 : light 1
        -1 : dark 1
        2 : light 2
        -2 : dark 2
        3 : light 3
        -3 : dark 3
        4 : light 4
        -4 : dark 4
        * Second column is start time of that trial
        * Third column is end time of that trial

        Returns
        -------
        tuple of ndarrays
            baselines and trials
        """
        baselines = self._load_file_escobar(
            extension='light_baselines_intervals')
        trials = self._load_file_escobar(
            extension='light_trials_intervals'
        )
        sampling_rate = self.get_sampling_rate_escobar()
        for a in [baselines, trials]:
            # Replace the string with the corresponding integer
            for s, i in str_to_int_escobar.items():
                a[a == s] = i
            # Convert all the times to seconds
            a[:, 1:] /= sampling_rate
        # Convert the arrays to type float.
        return baselines.astype(dtype=np.float64), trials.astype(
            dtype=np.float64)

    def _load_file_escobar(self, extension):
        """Load the time interval files from the Escobar 2016 data"""
        self.filename = self.params['dat']['filename']
        publication = self.params['dat']['publication']
        topdir = self.filename.split('-', 1)[0]
        self.path = os.path.join(MAINDIR, PATHS_DICTIONARY[publication], topdir,
                                 self.filename)
        fname = self.get_fname_from_extension(extension)
        return pd.read_csv(fname, delimiter=" ", header=None, usecols=[1,2,3]
                           ).as_matrix()

    def trial_intervals_and_gridscore_stats(self, trial, mode, stats='mean'):
        """
        Single grid score value for each interval of a trial

        Convenience function. See corresponding functions in plotting.py
        """
        # Get the grid scores for the given mode
        gridscores = self.computed['gridscores'][mode]
        # Get the spike time for the given mode
        spiketimes = self.get_spiketimes_in_mode(mode)
        # Get the time intervals for the trial type that is to be plotted
        intervals_trial = self.get_time_intervals_of_trial_type(trial)
        # Get the gridscores in the timewindows of that trial
        gs_in_intervals = self.get_gridscores_in_timewindows(gridscores,
                                                spiketimes, intervals_trial)
        gs_stats = self.get_stats_in_intervals(gs_in_intervals,
                                               stats=stats)
        return intervals_trial, gs_stats