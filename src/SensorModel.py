#!/usr/bin/env python

import numpy as np
import range_libc
import time

THETA_DISCRETIZATION = 112
INV_SQUASH_FACTOR = 2.2

Z_SHORT = 0.01
Z_MAX = 0.07
Z_RAND = 0.12
SIGMA_HIT = 8.0
Z_HIT = 0.75

class SensorModel:
	
  def __init__(self, map_msg, particles, weights):
    self.particles = particles
    self.weights = weights
  
    self.LASER_RAY_STEP = int(rospy.get_param("~laser_ray_step"))
    self.MAX_RANGE_METERS = float(rospy.get_param("~max_range_meters"))
    
    oMap = range_libc.PyOMap(map_msg)
    max_range_px = int(self.MAX_RANGE_METERS / map_msg.info.resolution)
    self.range_method = range_libc.PyCDDTCast(oMap, max_range_px, THETA_DISCRETIZATION)
    self.range_method.set_sensor_model(self.precompute_sensor_model(max_range_px))
    self.laser_angles = None
    self.downsampled_angles = None
    self.first_sensor_update = True
    self.do_resample = False
    
  def lidar_cb(self, msg):
    '''
    Initializes reused buffers, and stores the relevant laser scanner data for later use.
    '''
    if not isinstance(self.laser_angles, np.ndarray):
        self.laser_angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        self.downsampled_angles = np.copy(self.laser_angles[0::self.LASER_RAY_STEP]).astype(np.float32)

    self.downsampled_ranges = np.array(msg.ranges[::self.LASER_RAY_STEP])
    self.downsampled_ranges[np.isnan(self.downsampled_ranges)] = self.MAX_RANGE_METERS

    obs = (np.copy(self.downsampled_ranges).astype(np.float32), self.downsampled_angles)
    self.apply_sensor_model(self.particles, obs, self.weights)
    self.weights /= np.sum(self.weights)

    self.last_laser = msg
    self.do_resample = True
    
  def precompute_sensor_model(self, max_range_px):

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

  def apply_sensor_model(self, proposal_dist, obs, weights):
    '''
    This function computes a probablistic weight for each particle in the proposal distribution.
    These weights represent how probable each proposed (x,y,theta) pose is given the measured
    ranges from the lidar scanner.
    There are 4 different variants using various features of RangeLibc for demonstration purposes.
    - VAR_REPEAT_ANGLES_EVAL_SENSOR is the most stable, and is very fast.
    - VAR_NO_EVAL_SENSOR_MODEL directly indexes the precomputed sensor model. This is slow
    but it demonstrates what self.range_method.eval_sensor_model does
    '''
        
    obs_ranges = obs[0]
    obs_angles = obs[1]
    num_rays = obs_angles.shape[0]
    # only allocate buffers once to avoid slowness
    if self.first_sensor_update:
      self.queries = np.zeros((proposal_dist.shape[0],3), dtype=np.float32)
      self.ranges = np.zeros(num_rays*proposal_dist.shape[0], dtype=np.float32)
      self.first_sensor_update = False

    self.queries[:,:] = proposal_dist[:,:]

    self.range_method.calc_range_repeat_angles(self.queries, obs_angles, self.ranges)

    # evaluate the sensor model on the GPU
    self.range_method.eval_sensor_model(obs_ranges, self.ranges, weights, num_rays, proposal_dist.shape[0])

    np.power(weights, INV_SQUASH_FACTOR, weights)

