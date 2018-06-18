# import string
import numpy as np
import matplotlib.pyplot as plt

# #############################
##########	Labels	##########
##############################

#### PWDA ####
time_pwda = r'$\frac{a^2}{D_{\mathrm{hot}}}$'
time_label_pwda = r'$t/\frac{a^2}{D_{\mathrm{hot}}}$'
time_label_pwda_e4 = r'$t/ (10^4 \frac{a^2}{D_{\mathrm{hot}}})$'
time_label_pwda_e5 = r'$t/ (10^5 \frac{a^2}{D_{\mathrm{hot}}})$'
cluster_size_pwda = r'$M$'
Ncold = r'$N_{\mathrm{cold}}$'
Dcold = r'$D_{\mathrm{cold}}$'
packing_fraction_hot = r'$\phi_{\mathrm{hot}}$'
csd_avg = r'$\langle m_\mathrm{cold} \rangle / N_{\mathrm{cold}}$'
mean_max_cluster_size = r'$M_{\infty} / N_{\mathrm{cold}}$'
Deff = r'$D_{\mathrm{eff}}$'
Deff_Dhot = r'$D_{\mathrm{eff}} / D_{\mathrm{hot}}$'
Dcom_Dhot = r'$D_{\mathrm{com}} / D_{\mathrm{hot}}$'
Dcl_Dhot = r'$D_{\mathrm{cl}} / D_{\mathrm{hot}}$'
Dtilde = r'$D$'
M_over_Ncold = r'$M/$' + Ncold
tau_l = r'$\tau_{\mathrm{l}}$'
pair_correlation_function = r'$g(r)$'
r_over_a = r'$r/a$'
transition_diffconst = r'$D^*$'

#### Learning Grids ####
lrinh = r'$\eta_{\mathrm{I}}$'
ninh = r'$N_{\mathrm{I}}$'
ghinh_sq = r'$\alpha_{\mathrm{I}}^2$'
width_inh = r'$\sigma_{\mathrm{I}}$'
width_inh_m = r'$\sigma_{\mathrm{I}} [m]$'
width_inh_corr_m = r'$\sigma_{\mathrm{I,corr}} [m]$'
width_exc = r'$\sigma_{\mathrm{E}}$'
width_exc_corr = r'    $\sigma_{\mathrm{E,corr}}$'


######################################
##########	Color Cycles	##########
######################################
color_cycle_qualitative12 = [
	'#a6cee3',
	'#1f78b4',
	'#b2df8a',
	'#33a02c',
	'#fb9a99',
	'#e31a1c',
	'#fdbf6f',
	'#ff7f00',
	'#cab2d6',
	'#6a3d9a',
	'#ffff99',
	'#b15928',
]

color_cycle_qualitative_4 = ['#a6cee3','#1f78b4','#b2df8a','#33a02c']
color_cycle_qualitative10 = [
	'#A6CEE3', '#1F78B4', '#B2DF8A', '#33A02C', '#FB9A99',
	'#E31A1C', '#FDBF6F', '#FF7F00', '#CAB2D6', '#6A3D9A']
color_cycle_blue6 = [
	'#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#0c2c84']
color_cycle_blue4 = [
	'#253494', '#2C7FB8', '#41B6C4', '#A1DAB4']
color_cycle_blue3 = [
	'#225EA8', '#41B6C4', '#A1DAB4']
color_cycle_qualitative3 = [
	'#1f78b4', '#a6cee3', '#b2df8a']
color_cycle_red3 = [
	'#CC4C02', '#FE9929', '#FED98E']
color_cycle_red4 = [
	'#e31a1c', '#fd8d3c', '#fecc5c', '#ffffb2']
color_cycle_red5 = [
	'#bd0026', '#f03b20', '#fd8d3c', '#fecc5c', '#ffffb2']
color_cycle_qualitative3_2 = ['#5e3c99', '#e66101', '#fdb863']
color_cycle_pink3 = ['#ae017e', '#f768a1', '#fbb4b9']
######################################
##########	Marker Cycles	##########
######################################
marker_cycle = ['o', 's', '^', 'd', '*', 'p']
linestyle_cycle = ['-', '--', ':', '-.', '*', 'p']

def get_prefix(function_kwargs):
	"""
	Returns string of plotted functions for prefix in image names

	There is a lookup table to change plotting function names
	to something shorter in order to keep the redsulting
	image names short.


	Parameters
	----------
	function_kwargs : list
		A list of tuples like:
		[
		('plot_output_rates_from_equation', {'time': t1, 'from_file': True}),
		('plot_correlogram', {'time': t2, 'from_file': True}),
		...
		]

	Returns
	-------
	prefix : string
	"""
	# The lookup table
	prefix_lookup = {'plot_output_rates_from_equation': 'output_rates',
					 'plot_correlogram': 'correlogram',
					 'plot_time_evolution': 'time_evo'}
	# We take every ocurring function name only once
	# Note: the function name is always the first element in a
	# function_kwargs element
	prefix_list = list(set([x[0] for x in function_kwargs]))
	# Replace with value from lookup table if it exists
	for n, p in enumerate(prefix_list):
		try:
			prefix_list[n] = prefix_lookup[p]
		except KeyError:
			pass
	# Join the list to a single string
	prefix = '_'.join(prefix_list)
	return prefix

def cm2inch(value):
	"""
	Converts centimeters to inches
	"""
	return value / 2.54

def get_limits_with_buffer(minimum, maximum, buffer=0.2):
	"""
	Minimum and maximum with a symmetric offset

	Parameters
	----------
	minimum, maximum : float
		Minimum and maximum of the data that is plotted
	buffer : float between 0 and 1
		Defines how big the offset is
	Returns
	-------
	limits : ndarray
	"""
	buff = buffer * (maximum - minimum)
	limits = np.array([minimum - buff, maximum + buff])
	return limits



def simpleaxis(ax):
	"""
	Creates an axis with spines only left and bottom

	Taken from example idn stackoverflow
	"""
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()

def invisible_axis(ax):
	"""
	Does not show the axis, but without actually turning it off
	"""
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['left'].set_visible(False)

def remove_all_ticks(ax):
	plt.setp(ax,
			 xticks=[], yticks=[])
	ax.tick_params(
		axis='both',       # changes apply to both axes
		which='both',      # both major and minor ticks are affected
		bottom='off',      # ticks along the bottom edge are off
		top='off',         # ticks along the top edge are off
		left='off',      # ticks along the bottom edge are off
		right='off',      # ticks along the bottom edge are off
		labelbottom='on') # labels along the bottom edge are on



def adjust_spines(ax, spines):
	"""
	Taken from: http://matplotlib.org/1.3.1/examples/pylab_examples/spine_placement_demo.html
	"""
	for loc, spine in list(ax.spines.items()):
		if loc in spines:
			spine.set_position(('outward', 10))  # outward by 10 points
			spine.set_smart_bounds(True)
		else:
			spine.set_color('none')  # don't draw spine

	# turn off ticks where there is no spine
	if 'left' in spines:
		ax.yaxis.set_ticks_position('left')
	else:
		# no yaxis ticks
		ax.yaxis.set_ticks([])

	if 'bottom' in spines:
		ax.xaxis.set_ticks_position('bottom')
	else:
		# no xaxis ticks
		ax.xaxis.set_ticks([])


def circles(x, y, s, c='b', ax=None, vmin=None, vmax=None, **kwargs):
	"""
	Make a scatter of circles plot of x vs y, where x and y are sequence
	like objects of the same lengths. The size of circles are in data scale.

	Parameters
	----------
	x,y : scalar or array_like, shape (n, )
		Input data
	s : scalar or array_like, shape (n, )
		Radius of circle in data scale (ie. in data unit)
	c : color or sequence of color, optional, default : 'b'
		`c` can be a single color format string, or a sequence of color
		specifications of length `N`, or a sequence of `N` numbers to be
		mapped to colors using the `cmap` and `norm` specified via kwargs.
		Note that `c` should not be a single numeric RGB or
		RGBA sequence because that is indistinguishable from an array of
		values to be colormapped.  `c` can be a 2-D array in which the
		rows are RGB or RGBA, however.
	ax : Axes object, optional, default: None
		Parent axes of the plot. It uses gca() if not specified.
	vmin, vmax : scalar, optional, default: None
		`vmin` and `vmax` are used in conjunction with `norm` to normalize
		luminance data.  If either are `None`, the min and max of the
		color array is used.  (Note if you pass a `norm` instance, your
		settings for `vmin` and `vmax` will be ignored.)

	Returns
	-------
	paths : `~matplotlib.collections.PathCollection`

	Other parameters
	----------------
	kwargs : `~matplotlib.collections.Collection` properties
		eg. alpha, edgecolors, facecolors, linewidths, linestyles, norm, cmap

	Examples
	--------
	a = np.arange(11)
	circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')

	License
	--------
	This code is under [The BSD 3-Clause License]
	(http://opensource.org/licenses/BSD-3-Clause)
	"""
	from matplotlib.patches import Circle
	from matplotlib.collections import PatchCollection
	import pylab as plt
	#import matplotlib.colors as colors

	if ax is None:
		ax = plt.gca()

	if isinstance(c,str):
		color = c     # ie. use colors.colorConverter.to_rgba_array(c)
	else:
		color = None  # use cmap, norm after collection is created
	kwargs.update(color=color)


	if np.isscalar(x):
		patches = [Circle((x, y), s),]
	elif np.isscalar(s):
		patches = [Circle((x_,y_), s) for x_,y_ in zip(x,y)]
	else:
		patches = [Circle((x_,y_), s_) for x_,y_,s_ in zip(x,y,s)]
	collection = PatchCollection(patches, **kwargs)

	if color is None:
		collection.set_array(np.asarray(c))
		if vmin is not None or vmax is not None:
			collection.set_clim(vmin, vmax)

	ax.add_collection(collection)
	ax.autoscale_view()
	return collection
