import math
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .arrays import get_maximapositions_maximavalues
from .arrays import find_nearest
from .spikes import Spikes
from .plot_settings import *
from .correlogram import get_correlation_2d
from .correlogram import Gridness
from matplotlib import gridspec
from .dictionaries import str_to_int_escobar
from .dictionaries import mode_arguments
from .plot_settings import trial_colors

ARGUMENTS_DEFAULT = dict(
    shell_limits_choice='automatic_single',
    neighborhood_size_as_fraction=0.1,
    threshold_difference_as_fraction=0.1,
    cut_off_position=0,
    maximum_index=1,
    bins=100,
    weights=None,
    gridscore_norm=None,
    compare_to_other_symmetries=True,
    std_threshold=None,
    n_symmetry=6,
)


def simpleaxis(ax):
    """
    Creates an axis with spines only left and bottom

    Taken from example idn stackoverflow
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def plot_list(fig, plot_list, automatic_arrangement=True):
    """
    Takes a list of lambda forms of plot functions and plots them such that
    no more than four rows are used to assure readability

    Note: Instead of lambda forms, you can also use functools.partial to
    pass the not yet evaluated functions.
    """
    n_plots = len(plot_list)
    # A title for the entire figure (super title)
    # fig.suptitle('Time evolution of firing rates', y=1.1)
    if automatic_arrangement:
        for n, p in enumerate(plot_list, start=1):
            # Check if function name contains 'polar'
            # is needed for the sublotting.
            # A silly hack, that only works if every function that should
            # use polar plotting actually has the string 'polar' in its
            # name.
            if 'polar' in str(p.func):
                polar = True
            else:
                polar = False
            if n_plots < 4:
                fig.add_subplot(n_plots, 1, n, polar=polar)
            # plt.locator_params(axis='y', nbins=4)
            # plt.ylabel('firing rate')
            else:
                fig.add_subplot(math.ceil(n_plots / 2.), 2, n, polar=polar)
            # plt.locator_params(axis='y', nbins=4)
            p()


class Plot:
    def __init__(self, tables=None, psps=[None],
                 spikepositions=None, arena_limits=None,
                 spiketimes=None,
                 noise=0,
                 latex=None,
                 frame_slice=slice(None, None, None),
                 spikeheaddirections=None):
        self.psps = psps
        self.cmap = getattr(mpl.cm, 'viridis')
        self.tables = tables
        if self.tables is not None:
            # Works only if single psp is given
            self.params_and_rawdata_is_given = False
            self.set_params_rawdata_computed(psps[0])
            self.rawdata = self.plotsnep.rawdata
            try:
                self.computed = self.plotsnep.computed
            except AttributeError:
                self.computed = None
            self.spikepositions = self.plotsnep.rawdata['spikepositions'][
                frame_slice]
            self.spiketimes = self.plotsnep.rawdata['spiketimes'][frame_slice]
            self.arena_limits = self.plotsnep.rawdata['arena_limits']
            try:
                self.spikeheaddirections = self.plotsnep.rawdata[
                    'spikeheaddirections']
            except:
                self.headdirections = None
        else:
            self.spikepositions = spikepositions
            self.arena_limits = arena_limits
            self.spiketimes = spiketimes
            self.spikeheaddirections = spikeheaddirections

        self.spikes = Spikes(positions=self.spikepositions,
                             arena_limits=self.arena_limits,
                             times=self.spiketimes)

    def set_params_rawdata_computed(self, psp, set_sim_params=False):
        import general_utils.snep_plotting
        self.plotsnep = general_utils.snep_plotting.Snep()
        self.plotsnep.tables = self.tables
        return self.plotsnep.set_params_rawdata_computed(psp=psp,
                                                         set_sim_params=set_sim_params)

    def test_plot(self):
        plt.plot(np.arange(5))

    def gridscore_time_evolution(self, method='automatic_single_stdt1',
                                 ylim_conventional=(-0.7, 1.5),
                                 ylim_psi=(0, 0.4), color=None,
                                 every_nth_spikenumber=500,
                                 n_spikes_label=True,
                                 legend=True, time_label=True,
                                 xlim=None,
                                 xticks=None):
        """
        The time evolution of grid scores.

        Parameters
        ----------
        method : str {'langston', 'automatic_single_stdt1', ...}
            Method of the grid score whos time evolution is plotted.
            See computed.py for more.
        ylim_conventional : 2-tuple
            The y limits for conventional grid scores
        ylim_psi : 2-tuple
            The y limits for PSI based grid scores
        every_nth_spikenumber : int
            Multiple at which the number of spikes are indicated.
        Returns
        -------
        """
        color_psi = color_cycle_blue3[0]
        gridscores = self.computed['gridscores'][method]
        ax1 = plt.gca()
        ###############################################
        #### ax2 shows Langston or sargolini score ####
        ###############################################
        if (('sargolini' in method) or ('langston' in method)):
            if color:
                color = color
            else:
                color = color_cycle_blue3[2]
            # Conventional scores have the y axis on the right and use
            # a different scale (ylim)
            ax3 = ax1.twinx()
            plt.setp(ax3,
                     ylim=ylim_conventional,
                     xlabel='Time [s]',
                     ylabel='Langston',
                     yticks=[0, 1]
                     )
            times = gridscores[:, 0]
            gs = gridscores[:, 1]
            ax3.plot(times, gs, label=time_evo_labels[method], lw=2,
                     linestyle='solid', color=color)
            if legend:
                ax3.legend(loc='upper left', frameon=False)
            for ytick in ax3.get_yticklabels():
                ytick.set_color(color)
            ax3.yaxis.label.set_color(color)
        else:
            ###############################################
            ##########  ax2 shows PSI grid score ##########
            ###############################################
            times = gridscores[:, 0]
            gs = gridscores[:, 1]
            ax1.plot(times, gs,
                     label=time_evo_labels[method], lw=2,
                     zorder=100,
                     color=color_psi)
            if legend:
                ax1.legend(loc='upper right', frameon=False)
            plt.setp(ax1,
                     xlim=[0, self.spiketimes[-1]],
                     ylim=ylim_psi,
                     ylabel=r'$\Psi$ score',
                     yticks=[0, 0.3],
                     )
            if xticks:
                plt.xticks(xticks)
            if time_label:
                ax1.set_xlabel('Time [s]')
            # ax1.locator_params(axis='x', tight=True, nbins=3)
            for ytick in ax1.get_yticklabels():
                ytick.set_color(color_psi)
            ax1.yaxis.label.set_color(color_psi)
            ###############################################
            ###### Top x-axis shows number of spikes ######
            ###############################################
            # The number of spikes is indicated on the x-axis at the top
            # NB: The experimental time and the number of spikes are not
            # linearly aligned.
            ax2 = ax1.twiny()
            ax2.set_xlim(ax1.get_xlim())
            spiketimes, spikenumbers = self.spikes.get_spiketimes_spikenumbers(
                every_nth=every_nth_spikenumber, include_zero=True
            )
            spikenumbers_in_hundreds = spikenumbers / 100
            spikenumbers_in_hundreds = spikenumbers_in_hundreds.astype(
                np.int64)
            plt.setp(ax2,
                     ylabel=r'$\Psi$ score',
                     xticks=spiketimes,
                     xticklabels=spikenumbers_in_hundreds
                     )
            if n_spikes_label:
                ax2.set_xlabel('# Spikes [in hundreds]')
            # ax2.locator_params(axis='x', tight=True, nbins=3)
            for xtick in ax2.get_xticklabels():
                xtick.set_color('0.4')
            ax2.xaxis.label.set_color('0.4')
            # Set the current axis to ax1 again, to allow for adding plots
            # without messing with the labels
            plt.sca(ax1)

    def _print_trial_time_information(self):
        print('Spike number closest to half the trial time:')
        center_idx = find_nearest(self.spiketimes, self.spiketimes[-1] / 2,
                                  ret='index')
        print(center_idx + 1)
        print('Spike time closest to half the trial time:')
        center_time = find_nearest(self.spiketimes, self.spiketimes[-1] / 2,
                                   ret='element')
        print(center_time)
        print('Total number of spikes:')
        print(len(self.spikepositions))

    ###########################################################################
    ################## Gridscore time evolution Escobar 2016 ##################
    ###########################################################################

    def get_gridscores_in_timewindows(self, gridscores, spiketimes,
                                      timewindows, start_time_shift=0,
                                      end_time_shift=0):
        """
        Gridscores within time windows.



        NB: This is different from _gridscores_in_timewindows in computed.py.
        See documentation there.

        Parameters
        ----------
        gridscores : ndarray of shape (N)
            The grid scores of N spikes
        spiketimes : ndarray of shape (N)
            The spike times of the N spikes
        timewindows : ndarray of shape (M, 2)
            M timewindows like [start_time, end_time]
        start_time_shift : float
        end_time_shift : float

        Returns
        -------
        gs_in_tw : list of ndarrays
            [gridscores_in_first_timewindow,
            gridscoreds_in_second_timewindow, ...]
        """
        gs_in_tw = []
        for tw in timewindows:
            condition = np.logical_and(tw[0] + start_time_shift <= spiketimes,
                                       spiketimes < tw[1] + end_time_shift)
            gridscores_within_window = gridscores[condition]
            gs_in_tw.append(gridscores_within_window)
        return gs_in_tw

    def get_spiketimes_in_mode(self, mode='in_l3d3'):
        """"Spike times of all spikes that are part of a mode"""
        # Get the time intervals for all trials in the mode
        intervals_mode = self.get_time_intervals_from_mode(mode)
        # Index into all spikes that are part of the mode
        idx_mode = self.spike_idx(intervals_mode)
        # Get the spiketimes of all spikes that are part of the mode
        return self.spiketimes[idx_mode]

    def plot_gridscore_in_escobar_trial(self, mode='in_l3d3',
                                        trial='l3'):
        # Get the grid scores for the given mode
        gridscores = self.computed['gridscores'][mode]
        # Get the spike time for the given mode
        spiketimes = self.get_spiketimes_in_mode(mode)
        # Get the time intervals for the trial type that is to be plotted
        intervals_trial = self.get_time_intervals_of_trial_type(trial)
        # Get the gridscores in the timewindows of that trial
        gs_in_tw = self.get_gridscores_in_timewindows(gridscores,
                                                spiketimes, intervals_trial)
        # Plot the time windows and the gridscores within
        self.plot_gridscore_in_intervals(intervals_trial, gs_in_tw,
                                         color=trial_colors[trial], label=trial)

    def get_stats_in_intervals(self, gridscores_in_intervals, stats='mean'):
        """
        Grid score statistics within time intervals

        Parameters
        ----------
        gridscores_in_intervals : list of length N with ndarrays of different
        size elements.

        Returns
        -------
        ret : ndarray of shape (N)
            The statistical value of the grid scores within each interval
        """
        gs_stats = []
        for gs in gridscores_in_intervals:
            gs_stats.append(self._stats(gs, stats=stats))
        return np.array(gs_stats)


    def plot_gridscore_in_intervals(self, intervals, gridscores_in_intervals,
                                    stats='mean', color='gray', label=None):
        """
        Plots the time intervals along x and the gridscore statistics along y

        Parameters
        ----------
        intervals : ndarray of shape (N, 2)
        gridscores_in_intervals : list of length N of one dimensional
                    ndarrays of different length.
        stats : {'mean'}
            How to plot the grid score in each interval.
        """
        simpleaxis(plt.gca())
        gs_stats = self.get_stats_in_intervals(gridscores_in_intervals,
                                               stats=stats)
        widths = intervals[:, 1] - intervals[:, 0]
        plt.bar(intervals[:, 0], gs_stats, width=widths, align='edge',
                color=color, label=label)
        # Plot settings
        plt.legend(loc='upper right',
                   # bbox_to_anchor=(1,1),
                   numpoints=1,
                   fontsize=12,
                   frameon=False,
                   handlelength=0.5, ncol=2)
        ax = plt.gca()
        plt.setp(ax,
                 xlabel='Time [s]',
                 ylabel=mean_gridscore_str + ' score')
        ax.grid(axis='y', linestyle='--', color='gray', lw=0.75)

    # def plot_gridscores_in_trial_intervals_averaged_over_cells(self, trial,
    #                                     mode, identifiers):
    #     # Iterate over cells (identifiers)
    #     means_in_intervals = []
    #     # self.computed_full
    #     for i in identifiers:
    #         self.computed['mean_gridscore_in_trial_intervals'][trial][mode]

    def plot_decay_escobar(self, mode='in_l3d3', trial='l3',
                           resolution=10, total_trial_length=120):
        """
        Plots the decay of the grid with high temporal resolution.

        Parameters
        ----------
        mode : str
        trial : str
            See other functions
        resolution : float
            Temporal resolution in seconds
        total_trial_length : float
            Duration of a single light on or off trial
        """
        # The intervals that divide a single trial length with given resolution
        time_blocks = self.get_time_blocks(resolution,
                                    total_trial_length=total_trial_length)
        gs_in_blocks = self.get_gridscores_in_timeblocks(trial, mode,
                                                         time_blocks)
        self.plot_gridscore_in_intervals(time_blocks, gs_in_blocks,
                                         color=trial_colors[trial])

    def get_gridscores_in_timeblocks(self, trial, mode, time_blocks):
        """
        Returns gridscores of spikes in time blocks within trials

        Specifically written to analyze Escobar 2016 data.

        Parameters
        ----------
        trial : str
            The trial, for which time slices are cropped.
            E.g., trial='l2' means that each l2 trial is separated into time
            blocks and the gridscores within each of the blocks (for all
            instances of the l2 trial type) are returned.
        mode : str (see documentation elsewhere)
        time_blocks : ndarray of shape (N, 2)
            Start and end time of the intervals.

        Returns
        -------
        gridscores_in_timeblocks : list of ndarray
            The list has N elements. Each element is an ndarray. Different
            ndarrays can have different length.
        """
        # Get the resolution from the width of the first time block
        resolution = time_blocks[1, 0] - time_blocks[0, 0]
        # Get the grid scores for the given mode
        gridscores = self.computed['gridscores'][mode]
        # Get the spike times for the given mode
        spiketimes = self.get_spiketimes_in_mode(mode)
        # Get the time intervals for the given trial type
        intervals_trial = self.get_time_intervals_of_trial_type(trial)
        # The grid scores of all spikes within the temporal blocks (for each
        # iteration of the specified trial)
        gs_in_blocks = []
        for start in time_blocks[:, 0]:
            # Get cropped intervals, so a temporal crop of each iteration of
            # the specified trial.
            cropped_intervals = self.crop_intervals(
                intervals_trial, start=start, length=resolution)
            # Get the gridscores in all of theses blocks
            gs_in_cropped_intervals = self.get_gridscores_in_timewindows(
                gridscores, spiketimes, cropped_intervals)
            # Concatenate them, because they correspond to the same time block
            gs_in_blocks.append(
                np.concatenate(tuple(gs_in_cropped_intervals)))
        return gs_in_blocks

    def _stats(self, values, stats='mean'):
        """Convenience function to check different statistics"""
        if stats == 'mean':
            return np.mean(values)

    def plot_escobar_trials(self, mode='each_spike_final_l3d3', trials='l3'):
        """
        Plot gridscore of different escobar trials in a single plot

        See `plot_gridscore_in_escobar_trial`.
        """
        simpleaxis(plt.gca())
        for trial in np.atleast_1d(trials):
            self.plot_gridscore_in_escobar_trial(mode=mode, trial=trial)

    @staticmethod
    def get_time_blocks(resolution, total_trial_length=120):
        starts = np.arange(0, total_trial_length, resolution)
        ends = starts + resolution
        return np.hstack((starts[:, np.newaxis], ends[:, np.newaxis]))

    @staticmethod
    def crop_intervals(intervals, start=0, length=10):
        """
        Returns a temporal crop for each given interval

        Parameters
        ----------
        intervals : ndarray of shape (N, 2)
            Time intervals
        start : float
            The start time (relative to the start time in the interval)
        length : float
            The length of the cropping window
        Returns
        -------
        croped intervals : ndarray of shape (N, 2)
            A new intervals array

        Examples
        --------
        >>> intervals =  np.array([[0., 100.], [100., 200.], [300., 500.]])
        >>> start=10
        >>> length=10
        >>> plot.crop_intervals(intervals, start, length)
        np.array([[10., 20.], [110., 120.], [310., 320.]])

        """
        start_times =  intervals[:, 0, np.newaxis] + start
        end_times = intervals[:, 0, np.newaxis] + start + length
        return np.hstack((start_times, end_times))

    ### The older functions ###
    def gridscore_time_evolution_each_spike_final(self,
                                                  mode='each_spike_final'):
        color = color_cycle_blue3[0]
        # Get the grid scores for the given mode
        gridscores = self.computed['gridscores'][mode]
        # Get the spike times for the given mode
        spiketimes = self.get_spiketimes_in_mode(mode)
        # Plot the spike times and the grid score of the associated spike
        plt.plot(spiketimes, gridscores, color=color, linestyle='none',
                 marker='o', markersize=2)
        # Print information relevant for trials where the underlying
        # distribution was changed after half the trial time.
        # self._print_trial_time_information()

        # Plot settings
        ax = plt.gca()
        ymax = max([0.4, max(gridscores)]) * 1.05
        plt.setp(ax,
                 xlim=[0, self.spiketimes[-1]],
                 ylim=[0, ymax],
                 ylabel=r'$\psi_k$',
                 yticks=np.arange(0, 1, 0.1),
                 xticklabels=[]
                 )
        ax.grid(axis='y', linestyle='--', color='gray', lw=0.75)
        plt.locator_params(axis='y', tight=True, nbins=5)
        simpleaxis(ax)

    def _plot_bar_in_bin(self, timewindows, means, color=color_cycle_red3[0]):
        """
        Plots barplot of light and dark trials separated

        Also plots horizontal line for mean of all light and mean of all dark
        trials.

        Parameters
        ----------
        timewindows : ndarray of shape (N, 2)
            Time windows in which spikes were binned.
        means : ndarray of shape (N,)
            The mean grid score within each time window.
        trials : str {None, 'light', 'dark'}
            Selects which trials to show (a selection of the time windows).
            If None, all time windows are considered.
        """
        # if trial == 'light':
        #     # The time windows for the light trials are the first N/2
        #     slc = np.s_[:int(len(timewindows) / 2)]
        #     color = color_cycle_blue3[1]
        # elif trial == 'dark':
        #     # The time windows for the dark trials are the last N/2
        #     slc = np.s_[int(len(timewindows) / 2):]
        #     color = color_cycle_blue3[0]
        # else:
        #     slc = np.s_[:]
        #     color='black'
        tw = timewindows
        m = means
        twindow_starts = tw[:, 0]
        twindow_ends = tw[:, 1]
        widths = twindow_ends - twindow_starts
        plt.bar(twindow_starts, m, width=widths,
                align='edge', color=color, alpha=0.4, lw=0)
        mean_in_trial = np.mean(m)
        plt.axhline(y=mean_in_trial, color=color, lw=1,
                    linestyle='dotted')

    def gridscore_time_evolution_each_spike_final_binned(self,
                                                         mode='each_spike_final',
                                                         data=None,
                                                         n_bins=None):
        # Get the grid scores from mode
        gridscores = self.computed['gridscores'][mode]
        # Get the spike times from mode
        spiketimes = self.get_spiketimes_in_mode(mode)
        # Get the intervals of the given mdoe
        intervals = self.get_time_intervals_from_mode(mode)
        if data == 'escobar':
            timewindows = intervals
        else:
            timewindows = self.get_timewindows(n=n_bins,
                        last_spiketime=spiketimes[-1], windowmethod='bins')
        twindow_ends = timewindows[:, 1]

        gs_in_tw = self.get_gridscores_in_timewindows(gridscores,
                                                      spiketimes, timewindows)
        means = []
        for g in gs_in_tw:
            means.append(np.mean(g))
        means = np.array(means)

        self._plot_bar_in_bin(timewindows, means)
        # self._plot_bar_in_bin(timewindows, means, trial='dark')

        ymax = max([0.3, max(means)])
        ax = plt.gca()
        plt.setp(ax,
                 xlim=[0, self.spiketimes[-1]],
                 ylim=[0, ymax],
                 ylabel=r'$\langle \psi_k \rangle$',
                 yticks=[0, 0.1, 0.2, 0.3, 0.4],
                 # xticks = [0, 500],
                 xlabel='Time [s]'
                 )
        simpleaxis(ax)
        # Mark the ends of the time windows with vertical lines
        # plt.vlines(x=twindow_ends, ymin=0, ymax=ymax, color='gray',
        #            linestyle='--', lw=0.75)
        # Show a grid, to make comparison easier
        ax.grid(axis='y', linestyle='--', color='gray', lw=0.75)
        # Show the overall mean gridscore as a horizontal line
        mean_gridscore = np.mean(gridscores)
        mean_gs_color = 'black'
        plt.axhline(y=mean_gridscore, color=mean_gs_color, lw=1,
                    linestyle='solid')
        plt.text(self.spiketimes[-1] * 1.01, mean_gridscore, r'$\Psi$',
                 color=mean_gs_color, horizontalalignment='left',
                 verticalalignment='center')



    def gridscore_time_evolution_together(self, stepsize=1,
                                          gridscore_norm=None):
        # if gridscore_norm is 'all_neighbor_pairs':
        # 	self.gridscore_time_evolution(
        # method='automatic_single_neighbor_norm',
        # 								  stepsize=stepsize)
        # 	self.gridscore_time_evolution(
        # method='automatic_single_final_neighbor_norm',
        # 								  stepsize=1)
        # 	self.gridscore_time_evolution(
        # method='automatic_single_neighbor_norm_stdt1',
        # 								  stepsize=stepsize)
        # 	self.gridscore_time_evolution(
        # method='automatic_single_final_neighbor_norm_stdt1',
        # 								  stepsize=1)
        # elif gridscore_norm is None:
        # 	# self.gridscore_time_evolution(method='automatic_single',
        # 	# 							  stepsize=stepsize)
        # 	# self.gridscore_time_evolution(method='automatic_single_final',
        # 	# 							  stepsize=1)
        # 	self.gridscore_time_evolution(method='automatic_single_stdt1',
        # 								  stepsize=stepsize)
        # 	self.gridscore_time_evolution(
        # method='automatic_single_final_stdt1',
        # 								  stepsize=1)

        # self.gridscore_time_evolution(method='sargolini',
        # 							  stepsize=stepsize)

        ylim = (0.0, 1.0)
        # self.gridscore_time_evolution(method='window_300', ylim_psi=ylim)
        # self.gridscore_time_evolution(method='window_600', ylim_psi=ylim)
        # self.gridscore_time_evolution(method='window_1200', ylim_psi=ylim)
        # self.gridscore_time_evolution(method='decrease_from_left',
        # ylim_psi=ylim)
        # self.gridscore_time_evolution(method='decrease_from_left_langston',
        # ylim_psi=ylim)
        # self.gridscore_time_evolution(method='window_1200_langston',
        # ylim_psi=ylim)
        # self.gridscore_time_evolution(method='sargolini', ylim_psi=ylim)
        # self.gridscore_time_evolution(method='automatic_single',
        # ylim_psi=ylim)
        self.gridscore_time_evolution(method='window_300', ylim_psi=ylim)
        # self.gridscore_time_evolution(method='window_600', ylim_psi=ylim)
        # self.gridscore_time_evolution(method='decrease_from_left',
        # ylim_psi=ylim)
        self.gridscore_time_evolution(method='window_300_langston',
                                      ylim_psi=ylim, color='black')
        self.gridscore_time_evolution(method='langston', ylim_psi=ylim)

    def _get_psi_n(self, from_computed=False, return_shell_limits=False):
        """
        Returns psi_n

        Convenience function, that either computes psi or returns it from
        computed values in the .h5 file
        """

        shell_limits = self.spikes.get_shell_limits(
            shell_limits_choice=self.shell_limits_choice,
            neighborhood_size_as_fraction=self.neighborhood_size_as_fraction,
            cut_off_position=self.cut_off_position,
            threshold_difference_as_fraction=self
                .threshold_difference_as_fraction,
            bins=self.bins, weights=self.weights,
            maximum_index=self.maximum_index)

        if from_computed:
            psi_n = self.computed[
                'psi'][str(self.n_symmetry)][self.shell_limits_choice]['psi']
        else:
            psi_n = self.spikes.psi_n_all(
                n_symmetry=self.n_symmetry, shell_limits=shell_limits,
                normalization=self.gridscore_norm,
                std_threshold=self.std_threshold,
                compare_to_other_symmetries=self.compare_to_other_symmetries)
        if return_shell_limits:
            return psi_n, shell_limits
        else:
            return psi_n

    def set_nonspecified_attributes_to_default(self, attribute_default_dict):
        """
        Check if attributs (keys) are defined. If not, set to default (values).

        Parameters
        ----------
        attribute_default_dict : dict
            Keys are attributes of the class.
            Values are their default values.
        """
        for key, value in attribute_default_dict.items():
            try:
                getattr(self, key)
            except AttributeError:
                setattr(self, key, value)

    def get_spikes_class_for_spike_selection(self,
                                             n_spikes=None,
                                             idx=None):
        """
        Returns a Spikes class that only considers the first n_spikes
        or the spikes specified by an index array.

        Parameters
        ----------
        n_spikes : int
            The number of spikes that should be considered
        idx : Index array that selects spikes

        Returns
        -------
        sc : A Spikes instance
        """
        if idx is None:
            frame_slice = slice(None, n_spikes, None)
        else:
            frame_slice = idx
        try:
            sc = Spikes(positions=self.spikepositions[frame_slice],
                        arena_limits=self.arena_limits,
                        times=self.spiketimes[frame_slice])
        except AttributeError:
            # When used in experiment, there is no rawdata defined yet.
            # This is a hack now.
            # sc = Spikes(positions=self.spikepositions[frame_slice],
            # 				arena_limits=self.arena_limits['arena_limits'],
            # 				times=self.spiketimes['spiketimes'][frame_slice])
            return self.spikes
        return sc

    def gridscore_vs_location(self, shell_limits_choice='automatic_single',
                              neighborhood_size_as_fraction=0.1,
                              cut_off_position=0,
                              threshold_difference_as_fraction=0.1,
                              from_computed=False,
                              maximum_index=1):

        self.shell_limits_choice = shell_limits_choice
        self.neighborhood_size_as_fraction = neighborhood_size_as_fraction
        self.cut_off_position = cut_off_position
        self.threshold_difference_as_fraction = \
            threshold_difference_as_fraction
        self.maximum_index = maximum_index

        self.set_nonspecified_attributes_to_default(ARGUMENTS_DEFAULT)

        psi_n, shell_limits = self._get_psi_n(from_computed,
                                              return_shell_limits=True)
        psi_abs = np.absolute(psi_n)

        gridscore_in_parts = [
            np.mean(
                psi_abs[self._idx_in_part_of_arena(
                    part='x_range', x_range=[i, i + 0.5])])
            for i in [0, 0.5]
        ]
        n = np.array([0, 1])
        width = 0.5
        left = n + 1 - width / 2.
        plt.bar(left, gridscore_in_parts, width=width,
                color=color_cycle_blue3[0])
        ax = plt.gca()
        plt.setp(ax, xticks=[1, 2],
                 xticklabels=['left', 'right'],
                 xlabel='Box side',
                 ylabel=r'$\Psi$ score')
        plt.locator_params(axis='y', tight=True, nbins=2)
        simpleaxis(ax)

    def _idx_in_part_of_arena(self, part='right_half', x_range=None):
        if part == 'right_half':
            separation = ((self.arena_limits[0, 0]
                           + self.arena_limits[0, 1]) / 2.)
            idx = np.argwhere(self.spikes.pos[:, 0] > separation)
        elif part == 'left_half':
            separation = ((self.arena_limits[0, 0]
                           + self.arena_limits[0, 1]) / 2.)
            idx = np.argwhere(self.spikes.pos[:, 0] <= separation)
        elif part == 'x_range':
            idx = np.argwhere(
                np.logical_and(
                    x_range[0] <= self.spikes.pos[:, 0],
                    self.spikes.pos[:, 0] <= x_range[1])
            )
        return idx

    def get_time_intervals_of_trial_type(self, trial='baseline_start'):
        """
        Returns time intervals for a trial type for Escobar 2016 data.

        One trial type can occur within several time intervals, so more than
        one interval might be returned.

        Parameters
        ----------
        trial : str {'baseline_start', 'baseline_end', 'l1', 'd1', ..., 'l4',
                    'd4'}
            The trial type

        Returns
        -------
        intervals : ndarray of shape (N, 2)
            If the trial type occurs only once (like 'baseline_start'):
            [[tstart0, tend0]]
            If the trial type occurs more than once (like 'l1'):
            [[tstart0, tend0], [tstart1, tend1], ...]
        """
        baselines = self.computed['baselines']
        trials = self.computed['trials']
        if trial == 'baseline_start':
            intervals = baselines[0, 1:]
        elif trial == 'baseline_end':
            intervals = self.computed['baselines'][1, 1:]
        else:
            i = str_to_int_escobar[trial]
            intervals = trials[trials[:, 0] == i][:, 1:]
        return np.atleast_2d(intervals)

    def concatenate_time_intervals(self, trials=None):
        """
        Concatenate time intevals of multiple trials

        Parameters
        ----------
        trials : list of trial strings
            E.g. ['l1', 'd1']

        Returns
        -------
        a : ndarray of shape (N, 2)
            Note that this leads to time intervals that do not go from
            earlier times to later times. For example, if dark and light
            tirals are intertwined, we would get something like. So time
            intervals later in the list might correspond to earlier times.
            E.g.: Say trials = ['l1', 'd1] and
            [[tstart_l1_0, tend_l1_0], [tstart_l1_1, tend_l1_1],
                    [tstart_d1_0, tend_d1_0, [start_d1_1, tend_d1_1]]
        """
        a = np.empty((0, 2), np.float64)
        for trial in trials:
            a = np.concatenate((a,  self.get_time_intervals_of_trial_type(trial)))
        return a

    def get_time_intervals_from_mode(self, mode):
        """Time intervals in which grid scores were computed for the mode"""
        # If mode is None or it contains 'in_', the grid score was computed
        # for all spikes, so no time intervals are needed.
        if mode is None or 'in_' in mode:
            return None
        else:
            try:
                trials = mode_arguments[mode]['trials']
                t_ints = self.concatenate_time_intervals(
                    trials=trials)
                return t_ints
            except KeyError:
                return None

    def spike_idx(self, n):
        """
        Index array into spike positions and spike times

        NB: This index arrays does not necessarily follow the temporal order
        of spikes.

        Parameters
        ----------
        n : {int, ndarray}
            If `n` is an integer, the first `n` spikes will be indexed.
            If `n` is a one dimensionals array, it is considered
            to be an index array already.
            If `n` is a two dimensioanl array, the rows contain time intervals.

        Returns
        -------
        idx : ndarray
            Index array into spike positions
        """
        if n is None:
            idx = np.arange(len(self.spikepositions))
        elif isinstance(n, int):
            # First n spikes
            idx = np.arange(n)
        elif isinstance(n, float):
            raise TypeError('n must be of type float or ndarray')
        else:
            dimensions = n.ndim
            if dimensions == 1:
                # n is already an index array
                idx = n
            elif dimensions == 2:
                # n contains time intervals
                # spiketimes must exist
                if self.spiketimes is None:
                    raise ValueError(
                        '''
                        Spiketimes is None, but spiketimes must be given, if n
                        contains time intervals
                        '''
                    )
                else:
                    st = self.spiketimes
                # Add spike indeces within all given intervals
                idx = np.array([], dtype=np.int64)
                for interval in n:
                    idx_in_interval = np.argwhere(
                        (interval[0] <= st) & (st < interval[1])
                    ).flatten()
                    idx = np.concatenate((idx, idx_in_interval))
        return idx

    def is_escobar_trial(self, spike_selection):
        """Check if spike selection is escobar trial"""
        if not spike_selection:
            return False
        else:
            for s in spike_selection:
                if s in ['l1', 'l2', 'l3', 'l4', 'd1', 'd2', 'd3', 'd4',
                         'baseline_start', 'baseline_end']:
                    return True

    def spikemap(self,
                 shell_limits_choice='automatic_single',
                 color_code='psi_abs',
                 neighborhood_size_as_fraction=0.1,
                 cut_off_position=0,
                 threshold_difference_as_fraction=0.05,
                 noise=0,
                 noise_type='gaussian',
                 dotsize=7,
                 from_computed=False,
                 gridscore_norm=None,
                 colorbar_range='automatic',
                 std_threshold=None,
                 spike_selection=None,
                 show_shell=False,
                 n_symmetry=6,
                 compare_to_other_symmetries=True,
                 psi_title=True,
                 bins=100,
                 weights=None,
                 colorbar_label=True,
                 maximum_index=1,
                 show_colorbar=True,
                 ):

        self.shell_limits_choice = shell_limits_choice
        self.neighborhood_size_as_fraction = neighborhood_size_as_fraction
        self.cut_off_position = cut_off_position
        self.threshold_difference_as_fraction = \
            threshold_difference_as_fraction
        self.gridscore_norm = gridscore_norm
        self.std_threshold = std_threshold
        self.n_symmetry = n_symmetry
        self.compare_to_other_symmetries = compare_to_other_symmetries
        self.bins = bins
        self.weights = weights
        self.maximum_index = maximum_index
        # Create an index array `idx` to select spikes
        if self.is_escobar_trial(spike_selection):
            n = self.concatenate_time_intervals(
                trials=np.atleast_1d(spike_selection))
        else:
            n = spike_selection
        idx = self.spike_idx(n=n)

        self.spikes = self.get_spikes_class_for_spike_selection(idx=idx)
        print('Number of spikes: ', len(self.spikes.pos[:, 0]))
        # if i == 0:
        self.add_noise(noise, type=noise_type)
        psi_n, shell_limits = self._get_psi_n(from_computed,
                                              return_shell_limits=True)
        print('shell limits: ', shell_limits)
        psi_abs = np.absolute(psi_n)
        # Each angle between two neighboring spikes is increased in psi
        # by a factor of n. To get the local orientation, we need to
        # undo this rotation by dividing by n_symmetry.
        # NB: np.angle returns angles in the range -Pi to +Pi.
        # So for n_symmetry = 6 the line below  leads to
        # angles between -30 and 30 degrees.
        psi_angle = (np.angle(psi_n, deg=True)) / n_symmetry

        # Sort everthing according to psi_abs values
        # Then better values are more visible
        sort_idx = np.argsort(psi_abs)
        # good_idx = sort_idx[int(np.floor(len(sort_idx) / 5)):]
        # idx = idx[good_idx]

        psi_abs = psi_abs[sort_idx]
        psi_angle = psi_angle[sort_idx]
        x = self.spikes.pos[:, 0]
        y = self.spikes.pos[:, 1]
        x = x[sort_idx]
        y = y[sort_idx]

        if color_code == 'psi_abs':
            # self.cmap = getattr(mpl.cm, 'viridis')
            c = psi_abs
        elif color_code == 'psi_angle':
            # self.cmap = getattr(mpl.cm, 'hsv')
            c = psi_angle
            colorbar_range = np.array([-180, 180]) / n_symmetry
        elif color_code == 'headdirection':
            c = np.rad2deg(self.spikeheaddirections)
            colorbar_range = np.array([-180, 180])
        # size_half = 30
        # center = -150
        # condition = np.logical_and(center - size_half < c, c < center +
        # 						   size_half)
        # c[condition] = -1000
        else:
            c = -np.ones_like(psi_abs)
            color_code = 'psi_abs'

        max_factor = 0.5 if gridscore_norm == 'all_neighbor_pairs' else 1
        # cmasked = np.ma.masked_where(c == 0, c)
        cnorm, cmap = self.get_colornorm_colormap(c, color_code,
                                                  max_factor=max_factor,
                                                  colorbar_range=colorbar_range)
        cmap.set_over('white')
        cmap.set_under('white')
        plt.scatter(
            x,
            y,
            c=c,
            s=dotsize, linewidths=0., edgecolors='none',
            norm=cnorm, cmap=cmap, alpha=1.0)
        plt.gca().set_aspect('equal')
        if color_code == 'psi_abs':
            mean = np.mean(c)
            std = np.std(c)
            cb_ticks = np.array([0, np.amax(c)])
            cb_format = '%.1f'
            cb_label = gridscore_str
            cb_label_orientation = 'horizontal'
            labelpad = -2
        elif color_code == 'psi_angle':
            high = 30
            low = -30
            c_without_zeros = c[c != 0]
            mean = stats.circmean(c_without_zeros, high=high, low=low)
            std = stats.circstd(c_without_zeros, high=high, low=low)
            cb_ticks = np.array([-30, 0, 30])
            cb_format = '%d'
            cb_label = orientation_str
            cb_label_orientation = 'horizontal'
            labelpad = -2
        elif color_code == 'headdirection':
            high = 180
            low = -180
            c_without_zeros = c[c != 0]
            mean = stats.circmean(c_without_zeros, high=high, low=low)
            std = stats.circstd(c_without_zeros, high=high, low=low)
            cb_ticks = np.array([-180, 0, 180])
            cb_format = '%d'
            cb_label = 'Head direction'
            cb_label_orientation = 'vertical'
            labelpad = -4

        if show_colorbar:
            self.cb = plt.colorbar(format=cb_format, ticks=cb_ticks)
            if colorbar_label:
                self.cb.set_label(cb_label, rotation=cb_label_orientation,
                                  labelpad=labelpad)

        norm_dict = {None: 'none', 'all_neighbor_pairs': 'alln'}
        th = 0. if std_threshold is None else std_threshold
        title = 'M:{0:.2f} SD:{1:.2f}T:{3:.2f}n:{4}N:{2}'.format(
            mean, std, norm_dict[gridscore_norm], th, n_symmetry)
        if psi_title:
            if color_code == 'psi_abs':
                if shell_limits_choice == 'automatic_single_for_bands':
                    title_str = r'$\Psi_2$ = {0:.2f}'
                elif shell_limits_choice == 'automatic_single_for_quadratic':
                    title_str = r'$\Psi_4$ = {0:.2f}'
                else:
                    title_str = mean_gridscore_str + ' = {0:.2f}'
                plt.title(title_str.format(mean), fontsize=12)
            elif color_code == 'psi_angle':
                if np.isnan(mean):
                    title = mean_orientation_str + r' = invalid'
                else:
                    title = (mean_orientation_str
                             + r' = {0:d}$^\circ$').format(int(round(mean)))
                plt.title(title, fontsize=12)
            elif color_code == 'headdirection':
                title = '$\mu = {0:.0f}$, $\sigma = {1:.0f}$'.format(mean, std)
                plt.title(title, fontsize=12)

        if show_shell:
            plt.title(title, fontsize=10)
            origin = (x[-1], y[-1])
            self.show_shell_limits(shell_limits,
                                   origin=origin)
        plt.setp(plt.gca(),
                 xlim=self.arena_limits[0],
                 ylim=self.arena_limits[1],
                 xticks=[],
                 yticks=[])

        # Revert all modifications to the spikes class
        self.spikes = self.get_spikes_class_for_spike_selection()

    def get_colornorm_colormap(self, c, color_code='psi_abs', max_factor=1,
                               colorbar_range='automatic'):
        if colorbar_range is 'automatic':
            if color_code == 'psi_abs':
                colornorm = mpl.colors.Normalize(0, np.amax(c) * max_factor)
            elif color_code == 'psi_angle':
                colornorm = mpl.colors.Normalize(-30, 30)
            elif color_code == 'headdirection':
                print(color_code)
                colornorm = mpl.colors.Normalize(-180, 180)
        else:
            colornorm = mpl.colors.Normalize(colorbar_range[0],
                                             colorbar_range[1])
        if color_code == 'psi_abs':
            cmap = getattr(mpl.cm, 'viridis')
        elif color_code == 'psi_angle':
            cmap = getattr(mpl.cm, 'hsv')
        elif color_code == 'headdirection':
            # import seaborn as sns
            # from matplotlib.colors import ListedColormap
            # cmap_from_seaborn = sns.color_palette("husl", 360)
            # cmap = ListedColormap(cmap_from_seaborn)
            cmap = getattr(mpl.cm, 'hsv')
        return colornorm, cmap

    def show_shell_limits(self, shell_limits, origin=(0, 0)):
        n_shells = self.spikes.get_number_of_shells(shell_limits)
        plt.plot([origin[0]], [origin[1]], marker='x', color='black')
        if n_shells == 0:
            pass
        elif n_shells == 1:
            for r, c in [(shell_limits[0], 'red'),
                         (shell_limits[1], 'red')]:
                circle = plt.Circle(origin,
                                    r, ec=c, fc='none', lw=2,
                                    linestyle='dashed')
                plt.gca().add_artist(circle)
        elif n_shells == 2:
            for r, c in [(shell_limits[0, 0], 'red'),
                         (shell_limits[0, 1], 'red')]:
                circle = plt.Circle((0, 0),
                                    r, ec=c, fc='none', lw=2,
                                    linestyle='dashed')
                plt.gca().add_artist(circle)
            for r, c in [(shell_limits[1, 0], 'black'),
                         (shell_limits[1, 1], 'black')]:
                circle = plt.Circle((0, 0),
                                    r, ec=c, fc='none', lw=2,
                                    linestyle='dashed')
                plt.gca().add_artist(circle)

    def ratemap(self, sigma=5, spike_selection=None):
        if self.is_escobar_trial(spike_selection):
            n = self.concatenate_time_intervals(
                trials=np.atleast_1d(spike_selection))
        else:
            n = spike_selection
        idx = self.spike_idx(n=n)

        self.spikes = self.get_spikes_class_for_spike_selection(idx=idx)
        rm = self.spikes.get_ratemap(sigma=sigma)
        np.save('test_ratemap.npy', rm)
        plt.imshow(rm.T, origin='lower left', cmap=self.cmap)
        plt.setp(plt.gca(),
        		 xticks=[], yticks=[])
        plt.colorbar()

    def heatmap(self, bins=100):
        hm = self.spikes.get_heatmap(bins=bins)
        plt.imshow(hm.T, origin='lower left', cmap=self.cmap)
        plt.colorbar()

    def distance_histogram(self, neighborhood_size_as_fraction=0.1,
                           cut_off_position=0,
                           threshold_difference_as_fraction=0.05, noise=0,
                           noise_type='gaussian', n_spikes=None, bins=100,
                           weights=None, maximum_index=1):
        self.spikes = self.get_spikes_class_for_spike_selection(
            n_spikes=n_spikes)
        self.add_noise(noise, type=noise_type)
        n, centers = self.spikes.get_distancehistogram_and_centers(bins=bins,
                                                                   weights=weights)
        self._plot_distance_histogram(n, centers,
                                      neighborhood_size_as_fraction
                                      =neighborhood_size_as_fraction,
                                      cut_off_position=cut_off_position,
                                      threshold_difference_as_fraction
                                      =threshold_difference_as_fraction,
                                      bins=bins,
                                      weights=weights,
                                      maximum_index=maximum_index
                                      )

    def _plot_distance_histogram(self, n,
                                 centers, neighborhood_size_as_fraction=0.1,
                                 cut_off_position=0,
                                 threshold_difference_as_fraction=0.05,
                                 bins=100,
                                 weights=None, maximum_index=1):
        width = centers[1] - centers[0]
        plt.bar(centers, n, align='center', width=width, color='black')
        self.plot_maxima(n, centers, threshold_difference_as_fraction,
                         neighborhood_size_as_fraction)
        shell_limits = self.spikes.get_shell_limits(
            shell_limits_choice='automatic_single',
            neighborhood_size_as_fraction=neighborhood_size_as_fraction,
            cut_off_position=cut_off_position,
            threshold_difference_as_fraction=threshold_difference_as_fraction,
            bins=bins, weights=weights, maximum_index=maximum_index)
        self.indicate_typical_distance(shell_limits)
        ax = plt.gca()
        plt.setp(ax,
                 xlabel='Distance',
                 ylabel='# pairs',
                 # xticks=[0, 1],
                 # yticks=ax.get_yticks()/1000
                 )
        ax.locator_params(axis='y', tight=True, nbins=5)
        ax.locator_params(axis='x', tight=True, nbins=5)
        simpleaxis(ax)
        plt.margins(0.1)

    def indicate_typical_distance(self, shell_limits):
        typical_distance = np.mean(shell_limits)
        if typical_distance is not None:
            plt.axvline(x=typical_distance, lw=2, color='gray')
            plt.axvline(x=typical_distance - typical_distance / 6.,
                        lw=1, color='gray', linestyle='--')
            plt.axvline(x=typical_distance + typical_distance / 6.,
                        lw=1, color='gray', linestyle='--')
        print('typical distance: ', typical_distance)

    def plot_maxima(self, n, centers, threshold_diff, neighborhood_size):
        maxima_positions, maxima_values = \
            get_maximapositions_maximavalues(
                x=centers, y=n,
                threshold_difference_as_fraction
                =threshold_diff,
                neighborhood_size_as_fraction=neighborhood_size)
        plt.plot(maxima_positions, maxima_values,
                 marker='o', color='gray', linestyle='none',
                 markersize=5)

    def add_noise(self, noise, type='gaussian'):
        if noise != 0:
            s_x, s_y = self.spikes.pos.shape
            if type == 'gaussian':
                pos_noise = noise * np.random.randn(s_x, s_y)
            elif type == 'uniform':
                pos_noise = (
                        2 * noise * np.random.random_sample((s_x, s_y)) -
                        noise)
            self.spikes.pos += pos_noise
        else:
            pass

    def grid_score_vs_noise(self, norm=None):
        gs_sargolini = np.array([
            0.91, 0.91, 0.85, 0.78, 0.46, -0.029,
            0.39, -0.086, -0.36, -0.60, -2.0, -1.1
        ])
        gs_psi = np.array([
            0.47, 0.471, 0.467, 0.46, 0.441, 0.402,
            0.42, 0.344, 0.278, 0.201, 0.0642, 0.0487
        ])
        gs_psi_no_shell = np.array([
            0.11, 0.11, 0.108, 0.104, 0.101, 0.0951,
            0.101, 0.0881, 0.0769, 0.0691, 0.061, 0.0608
        ])
        my_slice = slice(0, 12)
        noise = np.arange(12)
        plt.subplot(311)
        gs = self.get_normalized_gs(gs_sargolini, norm=norm)
        plt.plot(noise[my_slice], gs, color='red')
        plt.ylabel('GS Sargolini')
        # plt.ylim([-2, 2])
        plt.subplot(312)
        gs = self.get_normalized_gs(gs_psi, norm=norm)
        plt.plot(noise[my_slice], gs, color='blue')
        # plt.ylim([0, 1])
        plt.xlabel('Uniform noise')
        plt.ylabel('GS psi')
        plt.subplot(313)
        gs = self.get_normalized_gs(gs_psi_no_shell, norm=norm)
        plt.plot(noise[my_slice], gs, color='blue')
        # plt.ylim([0, 1])
        plt.xlabel('Uniform noise')
        plt.ylabel('GS psi, no shell')

    def get_normalized_gs(self, grid_scores, norm=None):
        """
        Return normalized gs

        Parameters
        ----------
        grid_score : (N,) ndarray
            Array with N grid scores
        norm : str
            None :
            'start_one': Normalizes it such that the start value is 1

        Returns
        -------
        normalized_gs : (N,) ndarray
        """
        if norm == 'start_one':
            normalized_gs = grid_scores / grid_scores[0]
        elif norm is None:
            normalized_gs = grid_scores
        return normalized_gs

    def correlogram(self, method=None, sigma=5, noise=0,
                    noise_type='gaussian', n_spikes=None,
                    show_doughnut=True, mode='same'):
        """
        Plots a correlogram

        Parameters
        ----------
        n_spikes : int, optional
            See get_spikes_class_for_spike_selection for documenttion

        Returns
        -------
        """
        self.spikes = self.get_spikes_class_for_spike_selection(
            n_spikes=n_spikes)
        self.add_noise(noise, type=noise_type)
        # rm = self.spikes.get_list_of_ratemaps(sigma=sigma)
        rm = self.spikes.get_ratemap(sigma=sigma)
        corr_spacing, a = get_correlation_2d(
            rm, rm, mode=mode)
        # np.save('/Users/simonweber/programming/workspace/gridscore
        # /example_correlogram', a)
        # plt.imshow(a.T, vmin=-1, vmax=1, origin='lower left', cmap=self.cmap)
        corr_limits = 1 if mode == 'same' else 2
        corr_linspace = corr_limits * np.linspace(
            self.arena_limits[0, 0],
            self.arena_limits[0, 1],
            corr_spacing)
        X, Y = np.meshgrid(corr_linspace, corr_linspace)
        V = np.linspace(-1.0, 1.0, 30)
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.contourf(X, Y, a.T, V, cmap=self.cmap)
        plt.colorbar(format='%.0f', ticks=np.array([-1, 1]))
        if method is not None:
            radius = (
                             self.spikes.arena_limits[0, 1]
                             - self.spikes.arena_limits[0, 0]) / 2.
            if mode == 'full':
                correlogram_radius = 2 * radius
            else:
                correlogram_radius = radius
            gridness = Gridness(
                a, radius=correlogram_radius, method=method,
                neighborhood_size=10, threshold_difference=0.2,
                n_contiguous=100)
            grid_score = gridness.get_grid_score()

            if show_doughnut:
                title = 'GS: {0:.2}, noise: {1}'.format(
                    grid_score, noise)
                for r, c in [(gridness.inner_radius, 'black'),
                             (gridness.outer_radius, 'black'),
                             (gridness.grid_spacing, 'white')]:
                    circle = plt.Circle((0, 0),
                                        r, ec=c, fc='none', lw=2,
                                        linestyle='dashed')
                    plt.gca().add_artist(circle)
            else:
                title = 'Lngstn = {0:.1f}'.format(grid_score)
            plt.title(title, fontsize=12)
            plt.xticks([])
            plt.yticks([])

    def dummy(self):
        plt.plot(np.arange(5))

    # pass

    def gridscore_distribution(self,
                               shell_limits_choice=None,
                               neighborhood_size_in_percent=0.1,
                               cut_off_position=0,
                               threshold_difference_in_percent=0.05,
                               from_computed=False,
                               gridscore_norm=None,
                               std_threshold=None,
                               n_spikes=None):
        # Reset the spikes class such that it only considers the first n spikes
        # self.spikes = self.get_spikes_class_for_spike_selection(
        # 	n_spikes=n_spikes)

        shell_limits = self.spikes.get_shell_limits(
            shell_limits_choice=shell_limits_choice,
            neighborhood_size_as_fraction=neighborhood_size_in_percent,
            cut_off_position=cut_off_position,
            threshold_difference_as_fraction=threshold_difference_in_percent)
        psi_n = self._get_psi_n(n_symmetry=6,
                                shell_limits_choice=shell_limits_choice,
                                from_computed=from_computed,
                                gridscore_norm=gridscore_norm,
                                std_threshold=std_threshold)
        psi_abs = np.absolute(psi_n)
        n, bins, patches = plt.hist(psi_abs, bins=20)
        plt.xlabel('Grid score')
        plt.ylabel('Frequency')
        plt.axvline(x=np.mean(psi_abs), color='black')
        plt.axvline(x=np.median(psi_abs), color='red')
        plt.axvline(x=bins[np.argmax(n)] + (bins[1] - bins[0]) / 2,
                    color='green')

    def headdirection_polar(self):
        hd = self.spikeheaddirections
        plt.hist(hd)

    @staticmethod
    def vector_head_direction_tuning(angles):
        """
        Sums up vectors and normalizes with number of vectors

        Parameters
        ----------
        anlges : ndarray
            Angles of the vectors

        Returns
        -------
        ret : float
            The vector based head direction tuning
        """
        vectors = np.array([
            np.cos(angles),
            np.sin(angles)
        ])
        s = np.sum(vectors, axis=1)
        return np.linalg.norm(s) / vectors.shape[1]

    @staticmethod
    def get_timewindows(n, last_spiketime, windowsize=None,
                        windowmethod='fixed_size'):
        """
        Returns timewindows for running averages in compute or for plotting

        NB: In `compute`, `gridscores_within_timewindows` computes the grid
        score for only the spikes within a time window
        In `plotting`, given grid scores (for spikes within whatever time
        window) we show the means within time intervals.

        Parameters
        ----------
        n : int
            Number of windows returned
        windowsize : float
            The width of the timewindow in seconds
        last_spiketime : float
            The time of the last spike
        windowmethod : str {'fixed_size', 'decrease_from_left'}
            How to choose the windows.
            For 'fixed_size' a window of the size given by *windowsize*
            is moved from left to right in *n* steps.
            For 'decrease_from_left', the window initially contains all
            spikes and is decreased from the left in *n* steps, until only
            the last spike remains.
            For 'incrase' the time window is increased from the left. Note,
            that the inital time window is not [0, 0], because this would
            be useless, but [0, last_spiketime/n] instead.

        Returns
        -------
        timewindows : ndarray of shape (n, 2)
            Array of n start and end times
        """
        if windowmethod is 'fixed_size':
            twindow_starts = np.linspace(0,
                                         last_spiketime - windowsize,
                                         n)
            twindow_ends = twindow_starts + windowsize
        elif windowmethod is 'decrease_from_left':
            twindow_starts = np.linspace(0, last_spiketime, n)
            twindow_ends = np.repeat(last_spiketime, n)
        elif windowmethod is 'increase':
            twindow_starts = np.repeat(0, n)
            twindow_ends = np.linspace(last_spiketime / n, last_spiketime, n)
        elif windowmethod is 'bins':
            binsize = last_spiketime / n
            twindow_starts = np.arange(n) * binsize
            twindow_ends = twindow_starts + binsize
        timewindows = np.dstack((twindow_starts, twindow_ends))[0]
        return timewindows

    @staticmethod
    def shift_timewindows_to_relative_timewindows(timewindows):
        """
        Subtract the start time from each time window

        Both from the start time and the end time (this is ensures by adding
        a new axis)
        """
        relative_timewindows = timewindows  - timewindows[:, 0, np.newaxis]
        return relative_timewindows

if __name__ == '__main__':
    # # Simple test data
    # angle0 = 0
    # angle1 = np.pi / 3
    # angle2 = 2 * angle1
    # angle3 = 3 * angle1
    # angle4 = 4 * angle1
    # angle5 = 5 * angle1
    # rho = 1
    # spikepositions = np.array(
    # 	[
    # 		[0, 0],
    # 		pol2cart(rho, angle0),
    # 		pol2cart(rho, angle1),
    # 		pol2cart(rho, angle2),
    # 		pol2cart(rho, angle3),
    # 		pol2cart(rho, angle4),
    # 		pol2cart(rho, angle5),
    # 	]
    # )
    # #----------------------------

    ###########################################################################
    ############################ Hafting 2005 data ############################
    ###########################################################################
    from .spikedata import SpikeData

    n = 4
    i = 1
    spikedata = SpikeData(
        filename='figure2d_trial1',
        identifier='rat11015_t1c1'
    )
    spikepositions = spikedata.get_spikepositions()

    figs_dir = '/Users/simonweber/Library/Mobile Documents/' \
               'com~apple~CloudDocs/Gridscore/figs/'
    # publication_dir = os.path.join(figs_dir, spikedata.publication)
    # filename_dir = os.path.join(publication_dir, spikedata.filename)
    # identifier_dir = os.path.join(filename_dir, spikedata.identifier)
    directory = os.path.join(figs_dir, spikedata.publication,
                             spikedata.filename,
                             spikedata.identifier)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # for d in [publication_dir, filename_dir, identifier_dir]:
    # 	os.mkdir(d)

    ### Spikes with color coded grid score ###
    # fig.add_subplot(n, 1, i)
    # i += 1
    fig = plt.figure()
    shell_limits = np.array([0.9, 1.1]) * 44.6
    bins = spikedata.arena_limits[0, 1] * 2
    sigma = 8.
    spikes = Spikes(spikepositions, arena_limits=spikedata.arena_limits)
    plot = Plot(spikes)

    ### Spike map ###
    plot.spikemap()
    plt.savefig(os.path.join(directory, 'spikemap.png'))
    fig.clf()

    ### Rate map ###
    # fig.add_subplot(n, 1, i)
    # i += 1
    plot.ratemap()
    plt.savefig(os.path.join(directory, 'ratemap.png'))
    fig.clf()

    ### Correlogram ###
    # fig.add_subplot(n, 1, i)
    # i += 1
    plot.correlogram(method='sargolini')
    plt.savefig(os.path.join(directory, 'correlogram.png'))
    fig.clf()

    # ### Distances histogram ###
    print('calculating distances')
    plot.distance_histogram()
    plt.savefig(os.path.join(directory, 'distance_histogram.png'))
    fig.clf()

### Rate map from gridcells library ###
# diameter = spikedata.arena_limits[0, 1] * 2.
# arena = SquareArena(diameter, Pair2D(1., 1.))
# ax = fig.add_subplot(
# 	212, projection="gridcells_arena", arena=arena)
# plt.subplot(n, 1, i)
# i+=1
# pos = Position2D(spikes.pos_x + 90, spikes.pos_y + 90, 1.0)
# fake_spiketimes = np.arange(spikedata.spiketimes.shape[0])

# rate_map = gc_analysis.spatialRateMap(
# 	fake_spiketimes, pos, arena, sigma=50.)
# ax.spikes(fake_spiketimes, pos)
