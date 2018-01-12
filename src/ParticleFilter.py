#!/usr/bin/env python

import rospy 
import numpy as np
import range_libc
import time

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap
from geometry_msgs.msg import PoseStamped, PoseArray, PoseWithCovarianceStamped, PointStamped

from SensorModel import precompute_sensor_model

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
		self.range_method.set_sensor_model(self.precompute_sensor_model())
		self.initialize_global()

    self.pose_pub      = rospy.Publisher("/pf/viz/inferred_pose", PoseStamped, queue_size = 1)
    self.particle_pub  = rospy.Publisher("/pf/viz/particles", PoseArray, queue_size = 1)

    self.laser_sub = rospy.Subscriber(rospy.get_param("~scan_topic", "/scan"), LaserScan, self.lidarCB, queue_size=1)
    self.odom_sub  = rospy.Subscriber(rospy.get_param("~odometry_topic", "/odom"), Odometry, self.odomCB, queue_size=1)
    self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.clicked_pose, queue_size=1)
    self.click_sub = rospy.Subscriber("/clicked_point", PointStamped, self.clicked_pose, queue_size=1)

	def get_omap(self):
    map_service_name = rospy.get_param("~static_map", "static_map")
    print("Getting map from service: ", map_service_name)
    rospy.wait_for_service(map_service_name)
    map_msg = rospy.ServiceProxy(map_service_name, GetMap)().map

    self.map_info = map_msg.info
    oMap = range_libc.PyOMap(map_msg)
    self.MAX_RANGE_PX = int(self.MAX_RANGE_METERS / self.map_info.resolution)
		self.range_method = range_libc.PyCDDTCast(oMap, self.MAX_RANGE_PX, self.THETA_DISCRETIZATION)

     # 0: permissible, -1: unmapped, 100: blocked
    array_255 = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))

    # 0: not permissible, 1: permissible
    self.permissible_region = np.zeros_like(array_255, dtype=bool)
    self.permissible_region[array_255==0] = 1

	def initialize_global(self):
    permissible_x, permissible_y = np.where(self.permissible_region == 1)
		step = len(permissible_x)/self.MAX_PARTICLES
		indices = np.arange(0, len(permissible_x), step)[:self.MAX_PARTICLES]

    permissible_states = np.zeros((self.MAX_PARTICLES,3))
    permissible_states[:,0] = permissible_x[indices]
    permissible_states[:,1] = permissible_y[indices]
    permissible_states[:,2] = np.random.random(self.MAX_PARTICLES) * np.pi * 2.0

    Utils.map_to_world(permissible_states, self.map_info)
    self.particles = permissible_states
    self.weights[:] = 1.0 / self.MAX_PARTICLES

if __name__ == '__main__':
