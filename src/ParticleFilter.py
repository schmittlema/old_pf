#!/usr/bin/env python

import rospy 
import numpy as np
import time
import utils as Utils
import tf.transformations
import tf

from vesc_msgs.msg import VescStateStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.srv import GetMap
from geometry_msgs.msg import PoseStamped, PoseArray, PoseWithCovarianceStamped, PointStamped

from ReSample import ReSampler
from SensorModel import SensorModel
from MotionModel import OdometryMotionModel, KinematicMotionModel

class ParticleFilter():

  def __init__(self):
    self.MAX_PARTICLES = int(rospy.get_param("~max_particles"))
    self.MAX_VIZ_PARTICLES = int(rospy.get_param("~max_viz_particles"))

    self.particle_indices = np.arange(self.MAX_PARTICLES)
    self.particles = np.zeros((self.MAX_PARTICLES,3))

    self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES)

    map_service_name = rospy.get_param("~static_map", "static_map")
    print("Getting map from service: ", map_service_name)
    rospy.wait_for_service(map_service_name)
    map_msg = rospy.ServiceProxy(map_service_name, GetMap)().map
    self.last_laser = None
    self.map_info = map_msg.info

     # 0: permissible, -1: unmapped, 100: blocked
    array_255 = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))
    # 0: not permissible, 1: permissible
    self.permissible_region = np.zeros_like(array_255, dtype=bool)
    self.permissible_region[array_255==0] = 1

    print "Initializing particles"
    self.initialize_global()
   
    print 'Creating publishers'
    self.pose_pub      = rospy.Publisher("/pf/ta/viz/inferred_pose", PoseStamped, queue_size = 1)
    self.particle_pub  = rospy.Publisher("/pf/ta/viz/particles", PoseArray, queue_size = 1)
    self.pub_tf = tf.TransformBroadcaster()
    self.pub_laser     = rospy.Publisher("/pf/ta/viz/scan", LaserScan, queue_size = 1)

    ''' HACK VIEW INITIAL DISTRIBUTION'''
    '''
    self.MAX_VIZ_PARTICLES = 1000
    while not rospy.is_shutdown():
      self.visualize()
      rospy.sleep(0.2)
    '''
    self.resample_type = rospy.get_param("~resample_type", "naiive")
    self.resampler = ReSampler(self.particles, self.weights)

    self.sensor_model = SensorModel(map_msg, self.particles, self.weights)
    self.laser_sub = rospy.Subscriber(rospy.get_param("~scan_topic", "/scan"), LaserScan, self.sensor_model.lidar_cb, queue_size=1)
    
    self.MOTION_MODEL_TYPE = rospy.get_param("~motion_model", "kinematic")
    if self.MOTION_MODEL_TYPE == "kinematic":
      self.motion_model = KinematicMotionModel(self.particles)
      self.motion_sub = rospy.Subscriber(rospy.get_param("~motion_topic", "/vesc/sensors/core"), VescStateStamped, self.motion_model.motion_cb, queue_size=1)
    elif self.MOTION_MODEL_TYPE == "odometry":
      self.motion_model = OdometryMotionModel(self.particles)
      self.motion_sub = rospy.Subscriber(rospy.get_param("~motion_topic", "/vesc/odom"), Odometry, self.motion_model.motion_cb, queue_size=1)
    else:
      print "Unrecognized motion model: "+self.MOTION_MODEL_TYPE
      assert(False)
    
    self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.clicked_pose_cb, queue_size=1)
    self.click_sub = rospy.Subscriber("/clicked_point", PointStamped, self.clicked_pose_cb, queue_size=1)

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
  
  def initialize_global(self):

    print "GLOBAL INITIALIZATION"
    # randomize over grid coordinate space

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
  
  
  def publish_tf(self,pose, stamp=None):
    """ Publish a tf for the car. This tells ROS where the car is with respect to the map. """
    if stamp == None:
      stamp = rospy.Time.now()

    self.pub_tf.sendTransform((pose[0],pose[1],0),tf.transformations.quaternion_from_euler(0, 0, pose[2]), 
               stamp , "/ta_laser", "/map")

  def expected_pose(self):
    # returns the expected value of the pose given the particle distribution
    return np.dot(self.particles.transpose(), self.weights)
    
  def clicked_pose_cb(self, msg):
    '''
    Receive pose messages from RViz and initialize the particle distribution in response.
    '''
    if isinstance(msg, PointStamped):
      self.initialize_global()
    elif isinstance(msg, PoseWithCovarianceStamped):
      self.initialize_particles_pose(msg.pose.pose)

  def initialize_particles_pose(self, pose):
    '''
    Initialize particles in the general region of the provided pose.
    '''
    print "SETTING POSE"
    print pose
    self.weights[:] = 1.0 / float(self.particles.shape[0])
    self.particles[:,0] = pose.position.x + np.random.normal(loc=0.0,scale=0.5,size=self.particles.shape[0])
    self.particles[:,1] = pose.position.y + np.random.normal(loc=0.0,scale=0.5,size=self.particles.shape[0])
    self.particles[:,2] = Utils.quaternion_to_angle(pose.orientation) + np.random.normal(loc=0.0,scale=0.4,size=self.particles.shape[0])

  def visualize(self):
    print 'Visualizing...'
    self.inferred_pose = self.expected_pose()
    self.publish_tf(self.inferred_pose, rospy.Time.now())
    
    if self.pose_pub.get_num_connections() > 0 and isinstance(self.inferred_pose, np.ndarray):
      ps = PoseStamped()
      ps.header = Utils.make_header("map")
      ps.pose.position.x = self.inferred_pose[0]
      ps.pose.position.y = self.inferred_pose[1]
      ps.pose.orientation = Utils.angle_to_quaternion(self.inferred_pose[2])
      self.pose_pub.publish(ps)

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

  def publish_particles(self, particles):
    pa = PoseArray()
    pa.header = Utils.make_header("map")
    pa.poses = Utils.particles_to_poses(particles)
    self.particle_pub.publish(pa)

if __name__ == '__main__':
  rospy.init_node("particle_filter", anonymous=True)
  pf = ParticleFilter()
  
  while not rospy.is_shutdown():
    rospy.spin_once()
    if pf.sensor_model.do_resample:
      pf.sensor_model.do_resample = False
      
      if pf.resample_type == "naiive":
        pf.resampler.resample_naiive()
      elif pf.resample_type == "low_variance":
        pf.resampler.resample_low_variance()
      else:
        print "Unrecognized resampling method: "+ pf.resample_type      
      
      pf.visualize()



