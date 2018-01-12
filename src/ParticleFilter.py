#!/usr/bin/env python

import rospy 
import numpy as np
import range_libc
import time

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap

class ParticleFilter():

	def __init__(self):
		self.LASER_RAY_STEP = int(rospy.get_param("~laser_ray_step"))
		self.MAX_PARTICLES = int(rospy.get_param("~max_particles"))
		self.INV_SQUASH_FACTOR = 1.0/float(rospy.get_param("~squash_factor"))
		self.MAX_RANGE_METERS = float(rospy.get_param("~max_range"))
		self.THETA_DISCRETIZATION = int(rospy.get_param("~theta_discretization"))

		self.particles = np.zeros((self.MAX_PARTICLES,3))
		self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES)

		self.get_omap()

	def get_omap(self):

	

if __name__ == '__main__':
