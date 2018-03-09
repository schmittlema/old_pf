#!/usr/bin/env python

import numpy as np
import rospy
import range_libc
import time
from threading import Lock
from nav_msgs.srv import GetMap
import rosbag
import matplotlib.pyplot as plt
import utils as Utils
from sensor_msgs.msg import LaserScan

THETA_DISCRETIZATION = 112 # Discretization of scanning angle
INV_SQUASH_FACTOR = 0.5    # Factor for helping the weight distribution to be less peaked

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
    #self.range_method = range_libc.PyCDDTCast(oMap, max_range_px, THETA_DISCRETIZATION) # The range method that will be used for ray casting
    self.range_method = range_libc.PyRayMarchingGPU(oMap, max_range_px) # The range method that will be used for ray casting
    self.range_method.set_sensor_model(self.precompute_sensor_model(max_range_px)) # Load the sensor model expressed as a table
    self.queries = None
    self.ranges = None
    self.laser_angles = None # The angles of each ray
    self.downsampled_angles = None # The angles of the downsampled rays 
    self.do_resample = False # Set so that outside code can know that it's time to resample
    
  def lidar_cb(self, msg):
    '''
    Initializes reused buffers, and stores the relevant laser scanner data for later use.
    '''
    self.state_lock.acquire()
    #print 'in lidar_cb'
    if not isinstance(self.laser_angles, np.ndarray):
        #print 'Creating angles'
        self.laser_angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        self.downsampled_angles = np.copy(self.laser_angles[0::self.LASER_RAY_STEP]).astype(np.float32)

    #print 'Downsampling ranges'
    self.downsampled_ranges = np.array(msg.ranges[::self.LASER_RAY_STEP])
    self.downsampled_ranges[np.isnan(self.downsampled_ranges)] = self.MAX_RANGE_METERS

    # Compute the observation
    # obs is a a two element tuple
    # obs[0] is the downsampled ranges
    # obs[1] is the downsampled angles
    # Each element of obs must be a numpy array of type np.float32
    # Use self.LASER_RAY_STEP as the downsampling step
    # Keep efficiency in mind, including by caching certain things that won't change across future iterations of this callback
    #print 'Forming observations'
    obs = (np.copy(self.downsampled_ranges).astype(np.float32), self.downsampled_angles) 

        
    self.apply_sensor_model(self.particles, obs, self.weights)
    self.weights /= np.sum(self.weights)
    
    self.last_laser = msg
    self.do_resample = True
    #print 'FInished lidar cb'
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
    #print 'copying queries'
    self.queries[:,:] = proposal_dist[:,:]

    #print 'Raycasting'
    self.range_method.calc_range_repeat_angles(self.queries, obs_angles, self.ranges)

    #print 'Evaluating sensor model'
    # evaluate the sensor model on the GPU
    self.range_method.eval_sensor_model(obs_ranges, self.ranges, weights, num_rays, proposal_dist.shape[0])

    #print 'Squashing'

    np.power(weights, INV_SQUASH_FACTOR, weights)

if __name__ == '__main__':

  rospy.init_node("sensor_model", anonymous=True) # Initialize the node

  # Use the 'static_map' service (launched by MapServer.launch) to get the map
  map_service_name = rospy.get_param("~static_map", "static_map")
  print("Getting map from service: ", map_service_name)
  rospy.wait_for_service(map_service_name)
  map_msg = rospy.ServiceProxy(map_service_name, GetMap)().map # The map, will get passed to init of sensor model
  map_info = map_msg.info # Save info about map for later use    

  print 'Creating permissible region'
  # Create numpy array representing map for later use
  array_255 = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))
  permissible_region = np.zeros_like(array_255, dtype=bool)
  permissible_region[array_255==0] = 1 # Numpy array of dimension (map_msg.info.height, map_msg.info.width),
                                            # With values 0: not permissible, 1: permissible
  permissible_x, permissible_y = np.where(permissible_region == 1)
  
  # Potentially downsample permissible_x and permissible_y here
      
  print 'Creating particles'
  angle_step = 50
  particles = np.zeros((angle_step * permissible_x.shape[0],3))
  for i in xrange(angle_step):
    particles[i*(particles.shape[0]/angle_step):(i+1)*(particles.shape[0]/angle_step),0] = permissible_y[:]
    particles[i*(particles.shape[0]/angle_step):(i+1)*(particles.shape[0]/angle_step),1] = permissible_x[:]
    particles[i*(particles.shape[0]/angle_step):(i+1)*(particles.shape[0]/angle_step),2] = i*(2*np.pi / angle_step)
  
  Utils.map_to_world(particles, map_info)
  weights = np.ones(particles.shape[0])# / float(particles.shape[0])
  
  print 'Initializing sensor model'
  sm = SensorModel(map_msg, particles, weights)
  laser_sub = rospy.Subscriber(rospy.get_param("~scan_topic", "/scan"), LaserScan, sm.lidar_cb, queue_size=1)
  
  # Give time to get setup
  rospy.sleep(1.0)
  
  # Load laser scan from bag
  bag_path = '/home/bb8/racecar_ws/src/ta_lab1/bags/laser_scans/laser_scan1.bag'
  bag = rosbag.Bag(bag_path)
  for _, msg, _ in bag.read_messages(topics=['/scan']):
    laser_msg = msg
    break
  print 'angle_min = %f'%laser_msg.angle_min

  w_min = np.amin(weights)
  w_max = np.amax(weights)
  print 'w_min = %f'%w_min
  print 'w_max = %f'%w_max
  
  
  pub_laser = rospy.Publisher("/scan", LaserScan, queue_size = 1) # Publishes the most recent laser scan
  while not isinstance(sm.laser_angles, np.ndarray):
    print 'Publishing laser msg'
    pub_laser.publish(laser_msg)
    rospy.sleep(1.0)
 
  rospy.sleep(1.0) # Make sure there's enough time for laserscan to get lock
  
  print 'Going to wait for sensor model to finish'
  sm.state_lock.acquire()
  print 'Done, preparing to plot'
  weights = weights.reshape((angle_step, -1))
  
  weights = np.amax(weights, axis=0)
  print map_msg.info.height
  print map_msg.info.width
  print weights.shape
  w_min = np.amin(weights)
  w_max = np.amax(weights)
  print 'w_min = %f'%w_min
  print 'w_max = %f'%w_max
  weights = (weights-w_min)/(w_max-w_min)
  
  img = np.zeros((map_msg.info.height,map_msg.info.width))
  for i in xrange(len(permissible_x)):
    img[permissible_y[i],permissible_x[i]] = weights[i]
  plt.imshow(img)
