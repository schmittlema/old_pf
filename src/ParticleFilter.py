#!/usr/bin/env python

import rospy 
import numpy as np
import time
import utils as Utils
import tf.transformations
import tf
from threading import Lock

from vesc_msgs.msg import VescStateStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap
from geometry_msgs.msg import PoseStamped, PoseArray, PoseWithCovarianceStamped, PointStamped

from ReSample import ReSampler
from SensorModel import SensorModel
from MotionModel import OdometryMotionModel, KinematicMotionModel


class ParticleFilter():

  def __init__(self):
    self.MAX_PARTICLES = int(rospy.get_param("~max_particles")) # The maximum number of particles
    self.MAX_VIZ_PARTICLES = int(rospy.get_param("~max_viz_particles")) # The maximum number of particles to visualize

    self.particle_indices = np.arange(self.MAX_PARTICLES)
    self.particles = np.zeros((self.MAX_PARTICLES,3)) # Numpy matrix of dimension MAX_PARTICLES x 3
    self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES) # Numpy matrix containig weight for each particle

    self.state_lock = Lock() # A lock used to prevent concurrency issues. You do not need to worry about this

    # Use the 'static_map' service (launched by MapServer.launch) to get the map
    map_service_name = rospy.get_param("~static_map", "static_map")
    print("Getting map from service: ", map_service_name)
    rospy.wait_for_service(map_service_name)
    map_msg = rospy.ServiceProxy(map_service_name, GetMap)().map # The map, will get passed to init of sensor model
    self.map_info = map_msg.info # Save info about map for later use    

    # Create numpy array representing map for later use
    array_255 = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))
    self.permissible_region = np.zeros_like(array_255, dtype=bool)
    self.permissible_region[array_255==0] = 1 # Numpy array of dimension (map_msg.info.height, map_msg.info.width),
                                              # With values 0: not permissible, 1: permissible

    # Globally initialize the particles
    self.initialize_global()
   
    # Publish particle filter state
    self.pose_pub      = rospy.Publisher("/pf/ta/viz/inferred_pose", PoseStamped, queue_size = 1) # Publishes the expected pose
    self.particle_pub  = rospy.Publisher("/pf/ta/viz/particles", PoseArray, queue_size = 1) # Publishes a subsample of the particles
    self.pub_tf = tf.TransformBroadcaster() # Used to create a tf between the map and the laser for visualization
    self.pub_laser     = rospy.Publisher("/pf/ta/viz/scan", LaserScan, queue_size = 1) # Publishes the most recent laser scan

    self.pub_odom      = rospy.Publisher("/pf/ta/viz/odom", Odometry, queue_size = 1) # Publishes the path of the car
    ''' HACK VIEW INITIAL DISTRIBUTION'''
    '''
    self.MAX_VIZ_PARTICLES = 1000
    while not rospy.is_shutdown():
      self.visualize()
      rospy.sleep(0.2)
    '''
    self.RESAMPLE_TYPE = rospy.get_param("~resample_type", "naiive") # Whether to use naiive or low variance sampling
    self.resampler = ReSampler(self.particles, self.weights, self.state_lock)  # An object used for resampling

    self.sensor_model = SensorModel(map_msg, self.particles, self.weights, self.state_lock) # An object used for applying sensor model
    # Subscribe to laser scans. For the callback, use self.sensor_model's lidar_cb function
    self.laser_sub = rospy.Subscriber(rospy.get_param("~scan_topic", "/scan"), LaserScan, self.sensor_model.lidar_cb, queue_size=1)
    
    self.MOTION_MODEL_TYPE = rospy.get_param("~motion_model", "kinematic") # Whether to use the odometry or kinematics based motion model
    if self.MOTION_MODEL_TYPE == "kinematic":
      self.motion_model = KinematicMotionModel(self.particles, self.state_lock) # An object used for applying kinematic motion model
      # Subscribe to the state of the vesc (topic: /vesc/sensors/core). For the callback, use self.motion_model's motion_cb function
      self.motion_sub = rospy.Subscriber(rospy.get_param("~motion_topic", "/vesc/sensors/core"), VescStateStamped, self.motion_model.motion_cb, queue_size=1)
    elif self.MOTION_MODEL_TYPE == "odometry":
      self.motion_model = OdometryMotionModel(self.particles, self.state_lock)# An object used for applying odometry motion model
      # Subscribe to the vesc odometry (topic: /vesc/odom). For the callback, use self.motion_model's motion_cb function
      self.motion_sub = rospy.Subscriber(rospy.get_param("~motion_topic", "/vesc/odom"), Odometry, self.motion_model.motion_cb, queue_size=1)
    else:
      print "Unrecognized motion model: "+ self.MOTION_MODEL_TYPE
      assert(False)
    
    # Subscribe to the '/initialpose' topic. Publised by RVIZ. See clicked_pose_cb function in this file for more info
    self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.clicked_pose_cb, queue_size=1)
    #self.click_sub = rospy.Subscriber("/clicked_point", PointStamped, self.clicked_pose_cb, queue_size=1)
    print('Initialization complete')
  '''
  def initialize_global(self):
    permissible_x, permissible_y = np.where(self.permissible_region == 1)
    step = len(permissible_x)/self.particles.shape[0]
    indices = np.arange(0, len(permissible_x), step)[:self.particles.shape[0]]

    permissible_states = np.zeros((self.particles.shape[0],3))
    permissible_states[:,0] = permissible_y[indices]
    permissible_states[:,1] = permissible_x[indices]
    permissible_states[:,2] = np.random.random(self.particles.shape[0]) * np.pi * 2.0

    Utils.map_to_world(permissible_states, self.map_info)
    self.particles[:,:] = permissible_states[:,:]
    self.weights[:] = 1.0 / self.particles.shape[0]
  '''
  # Initialize the particles to cover the map
  def initialize_global(self):

    permissible_x, permissible_y = np.where(self.permissible_region == 1)
    
    angle_step = 4
    #indices = np.random.randint(0, len(permissible_x), size=self.particles.shape[0]/angle_step)
    step = 4*len(permissible_x)/self.particles.shape[0]
    indices = np.arange(0, len(permissible_x), step)[:(self.particles.shape[0]/4)]
    permissible_states = np.zeros((self.particles.shape[0],3))
    for i in xrange(angle_step):
      permissible_states[i*(self.particles.shape[0]/angle_step):(i+1)*(self.particles.shape[0]/angle_step),0] = permissible_y[indices]
      permissible_states[i*(self.particles.shape[0]/angle_step):(i+1)*(self.particles.shape[0]/angle_step),1] = permissible_x[indices]
      permissible_states[i*(self.particles.shape[0]/angle_step):(i+1)*(self.particles.shape[0]/angle_step),2] = i*(2*np.pi / angle_step)#np.random.random(self.particles.shape[0]) * np.pi * 2.0  
     
    Utils.map_to_world(permissible_states, self.map_info)
    self.particles[:,:] = permissible_states[:,:]
    self.weights[:] = 1.0 / self.particles.shape[0]
  
  # Publish a tf between the laser and the map
  # This is necessary in order to visualize the laser scan within the map
  def publish_tf(self,pose, stamp=None):
    """ Publish a tf for the car. This tells ROS where the car is with respect to the map. """
    if stamp == None:
      stamp = rospy.Time.now()

    self.pub_tf.sendTransform((pose[0],pose[1],0),tf.transformations.quaternion_from_euler(0, 0, pose[2]), 
               stamp , "/ta_laser", "/map")

  # Returns the expected pose given the current particles and weights
  def expected_pose(self):
    #print 'min, max, mean = %f, %f, %f'%(np.min(self.particles[:,2]), np.max(self.particles[:,2]), np.mean(self.particles[:,2]))
    cosines = np.cos(self.particles[:,2])
    sines = np.sin(self.particles[:,2])
    theta = np.arctan2(np.dot(sines,self.weights),np.dot(cosines, self.weights))
    position = np.dot(self.particles[:,0:2].transpose(), self.weights)
    return np.array((position[0], position[1], theta),dtype=np.float)
    
  # Callback for '/initialpose' topic. RVIZ publishes a message to this topic when you specify an initial pose using its GUI
  # Reinitialize your particles and weights according to the received initial pose
  # Remember to apply a reasonable amount of Gaussian noise to each particle's pose
  def clicked_pose_cb(self, msg):
    self.state_lock.acquire()
    pose = msg.pose.pose
    print "SETTING POSE"
    print pose
    self.weights[:] = 1.0 / float(self.particles.shape[0])
    self.particles[:,0] = pose.position.x + np.random.normal(loc=0.0,scale=0.5,size=self.particles.shape[0])
    self.particles[:,1] = pose.position.y + np.random.normal(loc=0.0,scale=0.5,size=self.particles.shape[0])
    self.particles[:,2] = Utils.quaternion_to_angle(pose.orientation) + np.random.normal(loc=0.0,scale=0.4,size=self.particles.shape[0])
    self.state_lock.release()
    
  # Visualize the current state of the filter
  # (1) Publishes a tf between the map and the laser. Necessary for visualizing the laser scan in the map
  # (2) Publishes the most recent laser measurement. Note that the frame_id of this message should be the child_frame_id of the tf from (1)
  # (3) Publishes a PoseStamped message indicating the expected pose of the car
  # (4) Publishes a subsample of the particles (use self.MAX_VIZ_PARTICLES). 
  #     Sample so that particles with higher weights are more likely to be sampled.
  def visualize(self):
    #print 'Visualizing...'
    self.state_lock.acquire()
    self.inferred_pose = self.expected_pose()
    self.publish_tf(self.inferred_pose, rospy.Time.now())
    
    if (self.pose_pub.get_num_connections() > 0 or self.pub_odom.get_num_connections() > 0) and isinstance(self.inferred_pose, np.ndarray):
      ps = PoseStamped()
      ps.header = Utils.make_header("map")
      ps.pose.position.x = self.inferred_pose[0]
      ps.pose.position.y = self.inferred_pose[1]
      ps.pose.orientation = Utils.angle_to_quaternion(self.inferred_pose[2])
      if(self.pose_pub.get_num_connections() > 0):
        self.pose_pub.publish(ps)
      if(self.pub_odom.get_num_connections() > 0):
        odom = Odometry()
        odom.header = ps.header
        odom.pose.pose = ps.pose
        self.pub_odom.publish(odom)

    if self.particle_pub.get_num_connections() > 0:
      if self.particles.shape[0] > self.MAX_VIZ_PARTICLES:
        # randomly downsample particles
        proposal_indices = np.random.choice(self.particle_indices, self.MAX_VIZ_PARTICLES, p=self.weights)
        # proposal_indices = np.random.choice(self.particle_indices, self.MAX_VIZ_PARTICLES)
        self.publish_particles(self.particles[proposal_indices,:])
      else:
        self.publish_particles(self.particles)
        
    if self.pub_laser.get_num_connections() > 0 and isinstance(self.sensor_model.last_laser, LaserScan):
      self.sensor_model.last_laser.header.frame_id = "/ta_laser"
      self.sensor_model.last_laser.header.stamp = rospy.Time.now()
      self.pub_laser.publish(self.sensor_model.last_laser)
    self.state_lock.release()

  def publish_particles(self, particles):
    pa = PoseArray()
    pa.header = Utils.make_header("map")
    pa.poses = Utils.particles_to_poses(particles)
    self.particle_pub.publish(pa)

# Suggested main 
if __name__ == '__main__':
  rospy.init_node("particle_filter", anonymous=True) # Initialize the node
  pf = ParticleFilter() # Create the particle filter
  
  while not rospy.is_shutdown(): # Keep going until we kill it
    # Callbacks are running in separate threads
    if pf.sensor_model.do_resample: # Check if the sensor model says it's time to resample
      pf.sensor_model.do_resample = False # Reset so that we don't keep resampling
      
      # Resample
      if pf.RESAMPLE_TYPE == "naiive":
        pf.resampler.resample_naiive()
      elif pf.RESAMPLE_TYPE == "low_variance":
        pf.resampler.resample_low_variance()
      else:
        print "Unrecognized resampling method: "+ pf.RESAMPLE_TYPE      
      
      pf.visualize() # Perform visualization



