import matplotlib as mpl
mpl.use('TkAgg')
import initialization
import plotting
import animating
import observables
import numpy as np
import matplotlib.pyplot as plt
import math
import time
# import functools
t0 = time.time()
##################################
##########	Parameters	##########
##################################
params = {
	'dimensions': 1,
	'boxtype': 'line',
	'boxlength': 1.0,
	'n_exc': 300,
	'n_inh': 300,
	'sigma_exc': 0.03,
	'sigma_inh': 0.1,
	'init_weight_noise_exc': 0.05,  # Noise on weights (as relative value)
	'init_weight_noise_inh': 0.05,  # 0.05 corresponds to +- 5 percent
	'simulation_time': 5.0,
	'target_rate': 5.0,
	'diff_const': 0.01,
	'dt': 1.0,
	# 'eta_exc': 0.000001,	
	'eta_exc': 0.000000000001,
	'eta_inh': 0.000000000002,
	'normalization': 'quadratic_multiplicative',
	# 'normalization': 'linear_multiplicative',
	# 'normalization': 'quadratic_multiplicative',
	'every_nth_step': 1,
	'seed': 2
}

params['init_weight_exc'] = 20. * params['target_rate'] / params['n_exc']
params['init_weight_inh'] = 5.0 * params['target_rate'] / params['n_inh']

params['initial_x'] = params['boxlength']/2.0
params['initial_y'] = params['boxlength']/2.0

np.random.seed(params['seed'])
######################################
##########	Initialize Rat	##########
######################################
rat = initialization.Rat(params)
##################################
#########	Move Rat	##########
##################################
rawdata = rat.run(output=True)
# print rawdata['exc_centers']
# print rawdata
##################################
##########	Plotting	##########
##################################
plot = plotting.Plot(params, rawdata)
fig = plt.figure()
# Create a list of lambda forms
# If you put multiple plot function calls into parentheses and in one
# lambda form, they get plotted on the same axis
plot_list = [
	lambda: [plot.fields_times_weights(syn_type='exc'), 
				plot.weights_vs_centers(syn_type='exc')],
	lambda: [plot.fields_times_weights(syn_type='inh'), 
				plot.weights_vs_centers(syn_type='inh')],
	# lambda: plot.weights_vs_centers(syn_type='exc', time=0/params['every_nth_step']),	
	# lambda: plot.weights_vs_centers(syn_type='inh', time=0/params['every_nth_step']),	
	# lambda: plot.weights_vs_centers(syn_type='exc', time=1/params['every_nth_step']),	
	# lambda: plot.weights_vs_centers(syn_type='inh', time=1/params['every_nth_step']),	
	# lambda: plot.weights_vs_centers(syn_type='exc', time=2/params['every_nth_step']),	
	# lambda: plot.weights_vs_centers(syn_type='inh', time=2/params['every_nth_step']),	
	# lambda: plot.weights_vs_centers(syn_type='exc', time=4/params['every_nth_step']),	
	# lambda: plot.weights_vs_centers(syn_type='inh', time=4/params['every_nth_step']),
	# lambda: plot.weights_vs_centers(syn_type='exc', time=8/params['every_nth_step']),	
	# lambda: plot.weights_vs_centers(syn_type='inh', time=8/params['every_nth_step']),	
	# lambda: plot.weights_vs_centers(syn_type='exc', time=16/params['every_nth_step']),	
	# lambda: plot.weights_vs_centers(syn_type='inh', time=16/params['every_nth_step']),
	# lambda: plot.weights_vs_centers(syn_type='exc', time=32/params['every_nth_step']),	
	# lambda: plot.weights_vs_centers(syn_type='inh', time=32/params['every_nth_step']),		
	# lambda: plot.weights_vs_centers(syn_type='exc', time=64/params['every_nth_step']),	
	# lambda: plot.weights_vs_centers(syn_type='inh', time=64/params['every_nth_step']),	
	# lambda: plot.weights_vs_centers(syn_type='exc', time=128/params['every_nth_step']),	
	# lambda: plot.weights_vs_centers(syn_type='inh', time=128/params['every_nth_step']),
	# lambda: plot.weights_vs_centers(syn_type='exc', time=256/params['every_nth_step']),	
	# lambda: plot.weights_vs_centers(syn_type='inh', time=256/params['every_nth_step']),
	# lambda: plot.weights_vs_centers(syn_type='exc', time=512/params['every_nth_step']),	
	# lambda: plot.weights_vs_centers(syn_type='inh', time=512/params['every_nth_step']),		
	# lambda: plot.weights_vs_centers(syn_type='exc', time=1024/params['every_nth_step']),	
	# lambda: plot.weights_vs_centers(syn_type='inh', time=1024/params['every_nth_step']),
	# lambda: plot.weights_vs_centers(syn_type='exc', time=2048/params['every_nth_step']),	
	# lambda: plot.weights_vs_centers(syn_type='inh', time=2048/params['every_nth_step']),
	# lambda: plot.weights_vs_centers(syn_type='exc', time=4096/params['every_nth_step']),	
	# lambda: plot.weights_vs_centers(syn_type='inh', time=4096/params['every_nth_step']),
	# lambda: plot.weights_vs_centers(syn_type='exc', time=8192/params['every_nth_step']),	
	# lambda: plot.weights_vs_centers(syn_type='inh', time=8192/params['every_nth_step']),
	# lambda: plot.weights_vs_centers(syn_type='exc'),	
	# lambda: plot.weights_vs_centers(syn_type='inh'),			
	# lambda: plot.weight_evolution(syn_type='exc'),
	# lambda: plot.weight_evolution(syn_type='inh'),
	# lambda: plot.output_rate_distribution(n_last_steps=63000),
	# # lambda: plot.output_rate_as_function_of_fields_and_weights(time=1000/params['every_nth_step']),
	# # lambda: plot.output_rate_as_function_of_fields_and_weights(time=2000/params['every_nth_step']),
	# # lambda: plot.output_rate_as_function_of_fields_and_weights(time=4000/params['every_nth_step']),	
	# # lambda: plot.output_rate_as_function_of_fields_and_weights(time=8000/params['every_nth_step']),
	lambda: plot.output_rate_as_function_of_fields_and_weights(time=0/params['every_nth_step']),
	lambda: [plot.output_rate_as_function_of_fields_and_weights(time=5/params['every_nth_step']),
				plot.output_rates_vs_positions()],			
			
	# lambda: plot.output_rate_as_function_of_fields_and_weights(time=62000/params['every_nth_step']),	
	# lambda: plot.output_rate_as_function_of_fields_and_weights(time=63000/params['every_nth_step']),
	# lambda: plot.output_rate_as_function_of_fields_and_weights(time=64000/params['every_nth_step']),	
	lambda: plot.output_rates_vs_positions(),
]
plotting.plot_list(fig, plot_list)

##################################
##########	Animating	##########
##################################
# save_path = '/Users/simonweber/Desktop/ani.mp4'
# animation = animating.Animation(rat, rawdata, start_time=0)
# animation.animate_all_synapses(interval=200)

print 'Plotting finished'
print time.time() - t0
plt.show()
