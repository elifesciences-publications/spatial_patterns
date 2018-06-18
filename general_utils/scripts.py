__author__ = 'simonweber'

import os
import subprocess
import numpy as np


def images2movies(maindir, framerate=10, delete_images=False,
				  overwrite=False, scale_flag='', extension='.png',
				  start_number=0, frames=int(1e8)):
	"""
	Creates mp4 movies from png files in each subdirectory of maindir

	Parameters
	----------
	maindir : str
		Directory with subdirectories that contain png files of the naming
		0.png 1.png 2.png ...
	framerate : int
		Frames per second of resulting movie
	scale_flag : str
		'-vf scale=1400:950'
		Sometimes ffmpeg has problems with rounding up the aspects of the input
		images. If this is the case give scale_flag a string
		string like '-vf scale=1400:950' where the two numbers are the rounded
		values of the pixels of the input images.
	start_number : int (opitional, default 0)
		The number of the first .png (or whatever extension) file.
		The numbers must rise in single integer steps.
		So typically you have file like 0.png, 1.png, 2.png that you combine 
		into a movie.
	frames : int (optional, default is very high, to not be constrained)
		Number of frames, starting at `start_number`
	"""
	subdirs = []
	for dirpath, dirnames, filenames in os.walk(maindir):
		subdirs.append(dirpath)

	subdirs = subdirs[1:]

	for sd in subdirs:
		# print sd
		cd_command = 'cd %s' % sd
		movie_name = os.path.basename(sd) + '_start_number_' + str(
			start_number)\
					 + '_frames_' + str(frames) + '.mp4'
		print(movie_name)
		ffmpeg_string = 'ffmpeg {0} -framerate {1} -start_number {5} -i %1d{2} -c:v libx264 {' \
						'3} -frames {6} ' \
						'-pix_fmt yuv420p {4}'

		# The -y flag overwriting existing videos with the same name
		if overwrite:
			overwrite_str = '-y'
		else:
			overwrite_str = ''
		ffmpeg_command = ffmpeg_string.format(overwrite_str, str(framerate),
											  extension,
											  scale_flag, movie_name,
											  start_number, frames)
		subprocess.call(cd_command + '; ' + ffmpeg_command, shell=True)
		if delete_images:
			subprocess.call('pwd', shell=True)
			subprocess.call(cd_command + '; rm *' + extension, shell=True)


def get_mean_firing_rate(dimensions, sigma, n_fields, radius):
	if dimensions == 1:
		area = 2 * radius
		exponent = 0.5
	elif dimensions == 2:
		area = 4 * radius**2
		exponent = 1

	mean_firing_rate = (
		n_fields * np.prod(sigma) * (2*np.pi)**exponent / area)
	return mean_firing_rate


if __name__ == '__main__':
	# maindir = '/Users/simonweber/localfiles/itb_experiments/' \
	# 		  'particles_with_different_activity/' \
	# 		  '2014-10-20-11h46m29s_1000_particles/visuals/videos'
	# maindir = '/Users/simonweber/localfiles/itb_experiments/' \
	# 		  'particles_with_different_activity/' \
	# 		  '2013-10-17-17h58m25s_Diffusive/visuals/videos'
	# maindir = '/Users/simonweber/localfiles/itb_experiments/learning_grids/2014-12-12-12h43m56s_nice_grid_video/visuals/videos/'
	# maindir = '/Users/simonweber/localfiles/itb_experiments/learning_grids/2015-01-15-17h05m43s/visuals/videos/output_rate/'
	# maindir = '/Users/simonweber/localfiles/itb_experiments/learning_grids/2015-01-05-17h44m42s_grid_score_stability/visuals/videos/rates_correlogram_2d/'
	# maindir = '/Users/simonweber/localfiles/itb_experiments/learning_grids/2015-02-17-11h22m18s/visuals/videos/rates_correlogram_2d/'
	# maindir = '/Users/simonweber/localfiles/itb_experiments/particles_with_different_activity/2013-10-17-17h58m25s_Diffusive/visuals/videos/'
	# maindir = '/Users/simonweber/localfiles/itb_experiments/particles_with_different_activity/2014-10-20-11h46m29s_1000_particles/visuals/videos_for_publication/'
	# maindir = '/Users/simonweber/localfiles/itb_experiments/particles_with_different_activity/2013-10-17-17h58m25s_Diffusive/visuals/videos_for_publication/'
	# maindir = '/Users/simonweber/localfiles/itb_experiments/learning_grids/2015-09-25-11h46m54s/'
	maindir ='/Users/simonweber/experiments/experiment_using_snep/2016-03-14-17h56m34s_GRF_2D_grid_stability/'
	images2movies(maindir=maindir, framerate=20, overwrite=True, delete_images=True,
				  scale_flag='-vf scale=584:584')
