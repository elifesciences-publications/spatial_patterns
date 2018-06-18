
import tables as tbls
import matplotlib as mpl
import matplotlib.pyplot as plt
import string
import os
import numpy as np
import functools
import sys
from six import iteritems
class Snep:
	"""Class providing SNEP related modules

	Classes that currently inherit from this one are:
		'Plot' in plotting.py (for different projects)

	Parameters
	----------
	params : snep parameters dictionary (needs to be given for automatic
				plotting in postproc)
	rawdata : snep rawdata dicitionary (needs to be given for automatic
				plotting in postproc)
	"""

	def __init__(self, params=None, rawdata=None, computed=None):
		if (params is not None) and (rawdata is not None):
			self.params = params
			self.rawdata = rawdata
			self.computed = computed
			self.params_and_rawdata_is_given = True
		else:
			self.params_and_rawdata_is_given = False
	def set_params_rawdata_computed(self, psp, set_sim_params=False):
		"""Uses tables from snep to set params, rawdata and computed values

		Parameters
		----------
		psp : snep paramspace point
		set_sim_params : bool
			If True: for every element in params['sim'] a self.variable is
					created (this can decrease the amount of writing and
					ensures downward compatibility)

		Returns
		-------

		"""

		if not self.params_and_rawdata_is_given:
			self.params = self.tables.as_dictionary(psp, True)
			# Downward compatibility
			# Set quantities that are newly defined to their old standard
			# value, if they are needee by plotting functions
			try:
				for syn_type in ['exc', 'inh']:
					if 'gaussian_height' not in self.params[syn_type]:
						self.params[syn_type]['gaussian_height'] = 1
			except KeyError:
				pass
			self.rawdata = self.tables.get_raw_data(psp)
			# Downward compatibility
			try:
				self.computed = self.tables.get_computed(psp)
			except:
				tbls.exceptions.NoSuchNodeError
		if set_sim_params:
			for k, v in list(self.params['sim'].items()):
				setattr(self, k, v)

def get_short_key_of_psps_key(k):
	"""Returns short version of a single psps key k"""
	for c in [' ', '(', ')', ',', ':', '\'']:
		k = str.replace(str(k), c, '')
	return k

def get_path_to_hdf_file(date_dir, project_name=None):
	if project_name == 'gridscore':
		ret = os.path.join(
			os.path.expanduser('~/experiments/gridscore_experiment/'),
			date_dir, 'experiment.h5')
	else:
		ret = os.path.join(
			os.path.expanduser('~/experiments/experiment_using_snep/'),
			date_dir, 'experiment.h5')
	return ret

def get_path_tables_psps(date_dir, project_name=None):
	"""
	Parameters
	----------
	date_dir : str
		The path to an h5 file within a project folder
	project_name : str
		This specifies the path of the project folder
		See: get_path_to_hdf_file
	Returns
	-------
	path : str
		Path to an .h5 experiment file
	tables : snep tables
	psps : list of parameter space points
	"""
	import snep.utils
	path = get_path_to_hdf_file(date_dir, project_name)
	tables = snep.utils.make_tables_from_path(path)
	tables.open_file(False)
	tables.initialize()
	print(tables)
	psps = tables.paramspace_pts()
	return path, tables, psps

def get_short_psps_dictionary(psps):
	"""Takes psps and returns a dictionary which can be used as a filename

	Graphics produced from several psps would have very long filenames, if all the
	psps are included. This function takes the psps and creates a small dictionary
	which contains all the psps that are plotted, but in a compact way. Unnecessary
	strings, whitespace and duplicates get removed.
	Parameters
	----------
	psps: paramspace points

	Returns
	-------
	short_psps: a dictionary with parameters as keys and list as values,
		the lists contain the ocurring values for the parameters, but
		without duplicates.
	"""
	short_psps = {}
	# Take the first psp to initialize the dictionary
	for k in psps[0]:
		# Get short version for each key
		short_key = get_short_key_of_psps_key(k)
		# Initialize dictionary
		short_psps[short_key] = []

	# Add each psp in psps
	for psp in psps:
		for k, v in iteritems(psp):
			short_key = get_short_key_of_psps_key(k)
			# Remove the uno and convert it back to float
			try:
				short_value = float(str.replace(str(v), ' uno', ''))
			except ValueError:
				short_value = str.replace(str(v), ' uno', '')
			# Append it to the corresponding list in the dictionary
			short_psps[short_key].append(short_value)
	# Remove possible duplicatdes (done by getting the set of v and
		# backtransforming to list)
	for k, v in iteritems(short_psps):
		short_psps[k] = list(set(v))
	return short_psps


def plot_psps(tables, psps, project_name, save_path=False, psps_in_same_figure=False,
		function_kwargs=None, prefix='filename', automatic_arrangement=True,
		file_type='pdf', latex=False, dpi=170, figsize=(5, 10),
			  transparent=False):
	"""Plot (several) paramspace paramspace_points

	Parameters
	----------
	tables : tables from an hdf5 file
	psps : list of paramspace_points (dictionaries)
	save_path : if specified, plots will be saved on disk
	psps_in_same_figure : if True, the given psps will all be plotted in the same figure
		otherwise a new plot for each psp in psps will be created
	function_kwargs : list of tuples of strings and dictionaries
		Structure: [(function_name1_string,
						{parameter11: value11, parameter12: value12, ...}),
					(function_name2_string),
						{parameter21: value21, parameter22: value22, ...}
					...
					]
	prefix : string
		Put at the beginning of the saved filename, to indicate what it
		stands for
	Returns
	-------
	None

	"""

	if project_name == 'learning_grids':
		import learning_grids.plotting as plotting
	elif project_name == 'gridscore':
		import gridscore.plotting as plotting
	elif project_name == 'particles_with_different_activity':
		import particles_with_different_activity.plotting as plotting
	elif project_name == 'attractor_network':
		import attractor_network.plotting as plotting
	else:
		print('ERROR: project name is unknown!')
		sys.exit(0)

	# mpl.rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
	# mpl.rc('text', usetex=True)

	if psps_in_same_figure:
		plot = plotting.Plot(tables, psps, latex=latex)
		fig = plt.figure(str(psps), figsize=figsize)
		# This creates a list of function with passed parameters and the
		# functions are ready to be called
		plot_list = [functools.partial(getattr(plot, f), **kwargs) for f, kwargs in function_kwargs]
		plotting.plot_list(fig, plot_list, automatic_arrangement)
		if save_path:
			short_psps = str(get_short_psps_dictionary(psps))
			print(short_psps)
			if len(short_psps) >= 255:
				short_psps = 'name_was_too_long'
			short_psps = (short_psps[:249] + '..') if len(short_psps) > 251 else short_psps
			save_path_full = os.path.join(save_path, prefix+'_'+short_psps + '.' + file_type)
			save_path_full = str.replace(save_path_full, ' ', '')
			save_path_full = str.replace(save_path_full, '\'', '')
			plt.savefig(save_path_full, dpi=dpi, bbox_inches='tight', pad_inches=0.01, transparent=transparent)
		# Clear figure and close windows
		else:
			plt.show()
		plt.clf()
		plt.close()

	else:
		for n, psp in enumerate(psps):
			print(n)
			print(psp)
			# Note that you have to pass a list to Plot
			plot = plotting.Plot(tables, [psp], latex=latex)
			fig = plt.figure(str(psp), figsize=figsize)
			# see above
			plot_list = [functools.partial(getattr(plot, f), **kwargs) for f, kwargs in function_kwargs]
			plotting.plot_list(fig, plot_list, automatic_arrangement)
			if save_path:
				# psp_str = str(psp)
				# psp_str = (psp_str[:249] + '..') if len(psp_str) > 251 else psp_str
				# save_path_full = os.path.join(save_path, string.replace(psp_str, ' uno', '') + '.pdf')
				save_path_full = os.path.join(save_path, prefix+'_'+tables.get_results_directory(psp)+'.' + file_type)
				# plt.savefig(save_path_full, dpi=170, bbox_inches='tight', pad_inches=0.1)
				# plt.tight_layout()
				# plt.savefig(save_path_full, dpi=100)
				plt.savefig(save_path_full, dpi=dpi, bbox_inches='tight', pad_inches=0.01, transparent=transparent)
			# Clear figure and close windows
			else:
				plt.show()
			plt.clf()
			plt.close()