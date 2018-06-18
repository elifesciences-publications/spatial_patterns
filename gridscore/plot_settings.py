######################################
##########	Color Cycles	##########
######################################
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
##########################################
################# Labels #################
##########################################

### Axis and colorbar label strings ###
gridscore_str = r'$|\psi|$'
orientation_str = r'$\theta$'
# mean_gridscore_str = r'$\langle |\psi| \rangle$'
mean_gridscore_str = r'$\Psi$'
mean_orientation_str = r'$\langle \theta \rangle$'

### Legend labels for time evolution of different grid score methods ###
time_evo_labels = dict(automatic_single='as',
					   automatic_single_final='asf',
					   automatic_single_neighbor_norm='asnn',
					   automatic_single_final_neighbor_norm='asfnn',
					   automatic_single_final_neighbor_norm_stdt1='asfnnt',
					   automatic_single_neighbor_norm_stdt1='asnnt',
					   automatic_single_stdt1='ast',
					   automatic_single_final_stdt1='asft',
					   sargolini='sargo',
					   window_600='w600',
					   window_600_langston='w600_langst',
					   window_300='w300',
					   window_1200='w1200',
					   decrease_from_left='decr_l',
					   decrease_from_left_langston='decr_l_langst',
					   window_1200_langston='w1200_langst',
					   window_300_langston='w300_langst',
					   langston='langst',
					   drop_recalc_1='drop_recalc_1',
					   compare_to_other_symmetries='cosy',
					   window_300_cosy='w300_cosy',
					   window_600_cosy='w600_cosy',
					   window_1200_cosy='w1200_cosy',
					   decrease_from_left_cosy=r'$\Psi$ from l.',
					   window_100_cosy_final='w100_cosy_f',
					   window_300_cosy_final='w300_cosy_f',
					   decrease_from_left_cosy_final='decr_l_cosy_f',
					   increase_cosy_final='cosy_f',
					   bins_cosy_final='bins_cosy_f'
					   )

### Colors for time evolution of different grid score measures ###
colors = dict(automatic_single=color_cycle_qualitative10[0],
			  automatic_single_final=color_cycle_qualitative10[1],
			  automatic_single_neighbor_norm=
			  color_cycle_qualitative10[2],
			  automatic_single_final_neighbor_norm=
			  color_cycle_qualitative10[3],
			  sargolini='red',
			  automatic_single_final_neighbor_norm_stdt1=
			  color_cycle_qualitative10[4],
			  automatic_single_stdt1=color_cycle_qualitative10[5],
			  automatic_single_final_stdt1=color_cycle_qualitative10[
				  6],
			  automatic_single_neighbor_norm_stdt1=
			  color_cycle_qualitative10[7],
			  window_300=color_cycle_qualitative10[8],
			  window_600=color_cycle_qualitative10[9],
			  window_1200=color_cycle_qualitative10[0],
			  decrease_from_left=color_cycle_blue6[0],
			  decrease_from_left_langston=color_cycle_blue6[1]
			  )

###############################################
######## Trial colors for Escobar 2016 ########
###############################################
color_light = color_cycle_blue4[2]
color_dark = color_cycle_blue4[0]
# color_light = 'yellow'
# color_light = '#ffd92f'
# color_dark = '0.2'
trial_colors = dict(
    l1=color_light, l2=color_light, l3=color_light, l4=color_light,
    d1=color_dark, d2=color_dark, d3=color_dark, d4=color_dark
)
