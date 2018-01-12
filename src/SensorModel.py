#!/usr/bin/env python

import numpy as np

Z_SHORT = 0.01
Z_MAX = 0.07
Z_RAND = 0.12
SIGMA_HIT = 8.0
Z_HIT = 0.75

def precompute_sensor_model(max_range_px)

	table_width = int(max_range_px) + 1
	sensor_model_table = np.zeros((table_width,table_width))

	# d is the computed range from RangeLibc
	for d in xrange(table_width):
		  norm = 0.0
		  sum_unkown = 0.0
		  # r is the observed range from the lidar unit
		  for r in xrange(table_width):
		      prob = 0.0
		      z = float(r-d)
		      # reflects from the intended object
		      prob += Z_HIT * np.exp(-(z*z)/(2.0*SIGMA_HIT*SIGMA_HIT)) / (SIGMA_HIT * np.sqrt(2.0*np.pi))

		      # observed range is less than the predicted range - short reading
		      if r < d:
		          prob += 2.0 * Z_SHORT * (d - r) / float(d)

		      # erroneous max range measurement
		      if int(r) == int(max_range_px):
		          prob += Z_MAX

		      # random measurement
		      if r < int(max_range_px):
		          prob += Z_RAND * 1.0/float(max_range_px)

		      norm += prob
		      sensor_model_table[int(r),int(d)] = prob

		  # normalize
		  sensor_model_table[:,int(d)] /= norm

	return sensor_model_table
