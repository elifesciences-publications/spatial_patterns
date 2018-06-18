import scipy
from pylab import *

# #############################################
# #########	Head Direction Tuning	##########
##############################################
class Head_Direction_Tuning():
	# """Class to get value of Head Direction tuning"""
	"""
	Class to get value of Head Direction tuning

	Parameters
	----------

	HD_firing_rates : ndarray
		Array with the firing rates of the output neuron along the head
		direction dimension.
		One dimensional of shape: (spacing)
	spacing : float
		The number of entries in `HD_firing_rates`

	Returns
	-------
	"""
	def __init__(self, HD_firing_rates, spacing, n=10000, alpha=0.05):
		self.HD_firing_rates = HD_firing_rates
		self.spacing = spacing
		self.n = n
		self.alpha = alpha


	def uniques_ties_cumuls_relfreqs(self, A):
		"""One liner description

		Parameters
		----------
		A : corresponds to directions in example

		Returns
		-------

		"""
		a = unique(A)
		# t: frequencies
		t = zeros(len(a))
		# m:
		m = zeros(len(a))
		for jj in range(len(a)):
			t[jj] = len(A[A == a[jj]])
			m[jj] = sum(t[:jj + 1])
		n = m[-1]
		m_n = m / n
		return a, t, m, n, m_n

	def watson_u2(self, ang1, ang2, alpha):
		"""
		adapted from pierre.megevand@gmail.com

		Computes Watson's U2 statistic for nonparametric 2-sample testing of
		circular data, accommodating ties. The code is derived from eq. 27.17
		Zar (1999) and its performance was verified using the numerical examples
		27.10 and 27.11 from that reference.
		Inputs:
		A1, A2:   vectors containing angles (in degrees or radians, the unit does
		not matter)

		Outputs:
		U2:       Watson's U2 statistic

		Significance tables for U2 have been published, e.g. in Zar (1999).
		Alternatively, an ad hoc permutation test can be used.

		References:
		Zar JH. Biostatistical Analysis. 4th ed., 1999.
		Chapter 27: Circular Distributions: Hypothesis Testing.
		Upper Saddle River, NJ: Prentice Hall.
		"""

		a1, t1, m1, n1, m1_n1 = self.uniques_ties_cumuls_relfreqs(ang1)
		a2, t2, m2, n2, m2_n2 = self.uniques_ties_cumuls_relfreqs(ang2)

		n = n1 + n2

		k = len(unique(append(ang1, ang2)))
		table = zeros((k, 4))
		table[:, 0] = unique(append(ang1, ang2))
		for ii in range(len(table)):
			if table[ii, 0] in a1:
				loc1 = find(a1 == table[ii, 0])[0]
				table[ii, 1] = table[ii, 1] + m1_n1[loc1]
				table[ii, 3] = table[ii, 3] + t1[loc1]
			else:
				if ii > 0:
					table[ii, 1] = table[ii - 1, 1]

			if table[ii, 0] in a2:
				loc2 = find(a2 == table[ii, 0])[0]
				table[ii, 2] = table[ii, 2] + m2_n2[loc2]
				table[ii, 3] = table[ii, 3] + t2[loc2]
			else:
				if ii > 0:
					table[ii, 2] = table[ii - 1, 2]

		#print table
		d = table[:, 1] - table[:, 2]
		t = table[:, 3]
		td = sum(t * d)
		td2 = sum(t * d ** 2)
		U2 = ((n1 * n2) / (n ** 2)) * (td2 - (td ** 2 / n))

		k1 = min(n1, n2)
		k2 = max(n1, n2)
		if k1 > 10:
			k1 = inf
		if k2 > 12:
			k2 = inf

		# Critical values (from Kanji, 100 statistical tests, 1999)
		len1 = array(
			[5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8,
			 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, inf])
		len2 = array(
			[5, 6, 7, 8, 9, 10, 11, 12, 6, 7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11,
			 12, 8, 9, 10, 11, 12, 9, 10, 11, 12, 10, 11, 12, inf])
		if alpha == 0.01:
			a = array(
				[nan, nan, nan, nan, 0.28, 0.289, 0.297, 0.261, nan, 0.282,
				 0.298, 0.262, 0.248, 0.262, 0.259, 0.304, 0.272, 0.255, 0.262,
				 0.253, 0.252, 0.250, 0.258, 0.249, 0.252, 0.252, 0.266, 0.254,
				 0.255, 0.254, 0.255, 0.255, 0.255, 0.268])
		elif alpha == 0.05:
			a = array(
				[0.225, 0.242, 0.2, 0.215, 0.191, 0.196, 0.19, 0.186, 0.206,
				 0.194, 0.196, 0.193, 0.19, 0.187, 0.183, 0.199, 0.182, 0.182,
				 0.187, 0.184, 0.186, 0.184, 0.186, 0.185, 0.184, 0.185, 0.187,
				 0.186, 0.185, 0.185, 0.185, 0.186, 0.185, 0.187])
		else:
			print(
				'Watson U2: This test is implemented for alpha levels of 0.05 and 0.01 only.')

		# Find critical value
		i1 = (len1 == k1)
		i2 = (len2 == k2)
		value = a[i1 * i2]
		if isnan(value):
			print("error: watson u2 test cannot be computed for given input")
		# Edit: Simon 13.11.2017, to handle this exception
		elif value.size == 0:
			return np.nan, False
		h = (U2 >= value)[0]

		return U2, h

	def draw_from_head_direction_distribution(self):
		xk = np.arange(self.spacing)
		pk = self.HD_firing_rates / np.sum(self.HD_firing_rates)
		hd_dist = scipy.stats.rv_discrete(
			a=0, b=self.spacing, name='hd_dist', values=(xk, pk))
		hd_angles = hd_dist.rvs(size=self.n) * 180 / self.spacing
		return hd_angles

	def get_watson_U2_against_uniform(self):
		uniform_angles = np.random.uniform(0, 180, self.n)
		hd_angles = self.draw_from_head_direction_distribution()
		return self.watson_u2(uniform_angles, hd_angles, self.alpha)
