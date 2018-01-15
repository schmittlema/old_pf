#!/usr/bin/env python

import numpy as np
import range_libc
import time
from threading import Lock

THETA_DISCRETIZATION = 112 # Discretization of scanning angle
INV_SQUASH_FACTOR = 2.2    # Factor for helping the weight distribution to be less peaked

Z_SHORT = 0.01  # Weight for short reading
Z_MAX = 0.07    # Weight for max reading
Z_RAND = 0.12   # Weight for random reading
SIGMA_HIT = 8.0 # Noise value for hit reading
Z_HIT = 0.75    # Weight for hit reading

class SensorModel:
	
  def __init__(self, map_msg, particles, weights, state_lock=None):
    if state_lock is None:
      self.state_lock = Lock()
    else:
      self.state_lock = state_lock
  
    self.particles = particles
    self.weights = weights
    
    self.LASER_RAY_STEP = int(rospy.get_param("~laser_ray_step")) # Step for downsampling laser scans
    self.MAX_RANGE_METERS = float(rospy.get_param("~max_range_meters")) # The max range of the laser
    
    oMap = range_libc.PyOMap(map_msg) # A version of the map that range_libc can understand
    max_range_px = int(self.MAX_RANGE_METERS / map_msg.info.resolution) # The max range in pixels of the laser
    self.range_method = range_libc.PyCDDTCast(oMap, max_range_px, THETA_DISCRETIZATION) # The range method that will be used for ray casting
    self.range_method.set_sensor_model(self.precompute_sensor_model(max_range_px)) # Load the sensor model expressed as a table
    self.laser_angles = None # The angles of each ray
    self.downsampled_angles = None # The angles of the downsampled rays 
    self.do_resample = False # Set so that outside code can know that it's time to resample
    
  def lidar_cb(self, msg):
    '''
    Initializes reused buffers, and stores the relevant laser scanner data for later use.
    '''
    self.state_lock.acquire(blocking=True)
    
    if not isinstance(self.laser_angles, np.ndarray):
        self.laser_angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        self.downsampled_angles = np.copy(self.laser_angles[0::self.LASER_RAY_STEP]).astype(np.float32)

    self.downsampled_ranges = np.array(msg.ranges[::self.LASER_RAY_STEP])
    self.downsampled_ranges[np.isnan(self.downsampled_ranges)] = self.MAX_RANGE_METERS

    # Compute the observation
    # obs is a a two element tuple
    # obs[0] is the downsampled ranges
    # obs[1] is the downsampled angles
    # Each element of obs must be a numpy array of type np.float32
    # Use self.LASER_RAY_STEP as the downsampling step
    # Keep efficiency in mind, including by caching certain things that won't change across future iterations of this callback
    
    obs = (np.copy(self.downsampled_ranges).astype(np.float32), self.downsampled_angles) 
    
    self.apply_sensor_model(self.particles, obs, self.weights)
    self.weights /= np.sum(self.weights)
    
    self.last_laser = msg
    self.do_resample = True
    
    self.state_lock.release()
    
  def precompute_sensor_model(self, max_range_px):

    table_width = int(max_range_px) + 1
    sensor_model_table = np.zeros((table_width,table_width))

    # Populate sensor model table as specified
    
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
        
    obs_ranges = obs[0]
    obs_angles = obs[1]
    num_rays = obs_angles.shape[0]
    # only allocate buffers once to avoid slowness
    if not isinstance(self.queries, np.ndarray):
      self.queries = np.zeros((proposal_dist.shape[0],3), dtype=np.float32)
      self.ranges = np.zeros(num_rays*proposal_dist.shape[0], dtype=np.float32)

    self.queries[:,:] = proposal_dist[:,:]

    self.range_method.calc_range_repeat_angles(self.queries, obs_angles, self.ranges)

    # evaluate the sensor model on the GPU
    self.range_method.eval_sensor_model(obs_ranges, self.ranges, weights, num_rays, proposal_dist.shape[0])

    np.power(weights, INV_SQUASH_FACTOR, weights)

