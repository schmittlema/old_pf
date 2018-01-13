#!/usr/bin/env python

import rospy 
import numpy as np
import time
import utils as Utils
import tf.transformations
import tf

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap
from geometry_msgs.msg import PoseStamped, PoseArray, PoseWithCovarianceStamped, PointStamped

from SensorModel import SensorModel
from OdometryModel import OdometryModel

MAX_RANGE_METERS = 5.6

class ParticleFilter():

  def __init__(self):
    self.LASER_RAY_STEP = int(rospy.get_param("~laser_ray_step"))
    self.MAX_PARTICLES = int(rospy.get_param("~max_particles"))
    self.MAX_VIZ_PARTICLES = int(rospy.get_param("~max_viz_particles"))

    self.laser_angles = None
    self.downsampled_angles = None

    self.particle_indices = np.arange(self.MAX_PARTICLES)
    self.particles = np.zeros((self.MAX_PARTICLES,3))

    self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES)

    self.last_pose = None
    self.last_stamp = None
    self.odometry_model = OdometryModel()

    map_service_name = rospy.get_param("~static_map", "static_map")
    print("Getting map from service: ", map_service_name)
    rospy.wait_for_service(map_service_name)
    map_msg = rospy.ServiceProxy(map_service_name, GetMap)().map
    self.sensor_model = SensorModel(map_msg)
    self.map_info = map_msg.info

     # 0: permissible, -1: unmapped, 100: blocked
    array_255 = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))
    # 0: not permissible, 1: permissible
    self.permissible_region = np.zeros_like(array_255, dtype=bool)
    self.permissible_region[array_255==0] = 1

    print "Initializing particles"
    self.initialize_global()

    print 'Creating publishers'
    self.pose_pub      = rospy.Publisher("/pf/viz/inferred_pose", PoseStamped, queue_size = 1)
    self.particle_pub  = rospy.Publisher("/pf/viz/particles", PoseArray, queue_size = 1)
    self.pub_tf = tf.TransformBroadcaster()
    print 'Creating subscribers'
    self.laser_sub = rospy.Subscriber(rospy.get_param("~scan_topic", "/scan"), LaserScan, self.lidarCB, queue_size=1)
    self.odom_sub  = rospy.Subscriber(rospy.get_param("~odometry_topic", "/odom"), Odometry, self.odomCB, queue_size=1)
    self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.clicked_pose_cb, queue_size=1)
    self.click_sub = rospy.Subscriber("/clicked_point", PointStamped, self.clicked_pose_cb, queue_size=1)

  '''
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
  '''
  def initialize_global(self):
    '''
    Spread the particle distribution over the permissible region of the state space.
    '''
    print "GLOBAL INITIALIZATION"
    # randomize over grid coordinate space

    permissible_x, permissible_y = np.where(self.permissible_region == 1)
    indices = np.random.randint(0, len(permissible_x), size=self.MAX_PARTICLES)

    permissible_states = np.zeros((self.MAX_PARTICLES,3))
    permissible_states[:,0] = permissible_y[indices]
    permissible_states[:,1] = permissible_x[indices]
    permissible_states[:,2] = np.random.random(self.MAX_PARTICLES) * np.pi * 2.0

    Utils.map_to_world(permissible_states, self.map_info)
    self.particles = permissible_states
    self.weights[:] = 1.0 / self.MAX_PARTICLES

  def publish_tf(self,pose, stamp=None):
    """ Publish a tf for the car. This tells ROS where the car is with respect to the map. """
    if stamp == None:
      stamp = rospy.Time.now()

    # this may cause issues with the TF tree. If so, see the below code.
    self.pub_tf.sendTransform((pose[0],pose[1],0),tf.transformations.quaternion_from_euler(0, 0, pose[2]), 
               stamp , "/laser", "/map")

  def lidarCB(self, msg):
    '''
    Initializes reused buffers, and stores the relevant laser scanner data for later use.
    '''
    if not isinstance(self.laser_angles, np.ndarray):
        self.laser_angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        print 'LASER_RAY_STEP = %d'%self.LASER_RAY_STEP
        self.downsampled_angles = np.copy(self.laser_angles[0::self.LASER_RAY_STEP]).astype(np.float32)

    self.downsampled_ranges = np.array(msg.ranges[::self.LASER_RAY_STEP])
    self.downsampled_ranges[np.isnan(self.downsampled_ranges)] = MAX_RANGE_METERS

    obs = (np.copy(self.downsampled_ranges).astype(np.float32), self.downsampled_angles)
    self.sensor_model.apply_sensor_model(self.particles, obs, self.weights)
    self.weights /= np.sum(self.weights)

    # RESAMPLE
    proposal_indices = np.random.choice(self.particle_indices, self.MAX_PARTICLES, p=self.weights)
    self.particles = self.particles[proposal_indices,:]    
    self.weights[:] = 1.0 / self.MAX_PARTICLES

    self.inferred_pose = self.expected_pose()
    self.last_stamp = rospy.Time.now()
    self.publish_tf(self.inferred_pose, self.last_stamp)
    self.visualize()

  def expected_pose(self):
    # returns the expected value of the pose given the particle distribution
    return np.dot(self.particles.transpose(), self.weights)

  def odomCB(self, msg):
    '''
    Store deltas between consecutive odometry messages in the coordinate space of the car.
    Odometry data is accumulated via dead reckoning, so it is very inaccurate on its own.
    '''
    position = np.array([msg.pose.pose.position.x,
		         msg.pose.pose.position.y])

    orientation = Utils.quaternion_to_angle(msg.pose.pose.orientation)
    pose = np.array([position[0], position[1], orientation])

    if isinstance(self.last_pose, np.ndarray):
      rot = Utils.rotation_matrix(-self.last_pose[2])
      delta = np.array([position - self.last_pose[0:2]]).transpose()
      local_delta = (rot*delta).transpose()

		  # changes in x,y,theta in local coordinate system of the car
      control = np.array([local_delta[0,0], local_delta[0,1], orientation - self.last_pose[2]])
      self.odometry_model.apply_motion_model(self.particles, control)


    self.last_pose = pose

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
    self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES)
    self.particles[:,0] = pose.position.x + np.random.normal(loc=0.0,scale=0.5,size=self.MAX_PARTICLES)
    self.particles[:,1] = pose.position.y + np.random.normal(loc=0.0,scale=0.5,size=self.MAX_PARTICLES)
    self.particles[:,2] = Utils.quaternion_to_angle(pose.orientation) + np.random.normal(loc=0.0,scale=0.4,size=self.MAX_PARTICLES)

  def visualize(self):
    print 'Visualizing...'
    if self.pose_pub.get_num_connections() > 0 and isinstance(self.inferred_pose, np.ndarray):
      ps = PoseStamped()
      ps.header = Utils.make_header("map")
      ps.pose.position.x = self.inferred_pose[0]
      ps.pose.position.y = self.inferred_pose[1]
      ps.pose.orientation = Utils.angle_to_quaternion(self.inferred_pose[2])
      self.pose_pub.publish(ps)

    if self.particle_pub.get_num_connections() > 0:
      if self.MAX_PARTICLES > self.MAX_VIZ_PARTICLES:
        # randomly downsample particles
        proposal_indices = np.random.choice(self.particle_indices, self.MAX_VIZ_PARTICLES, p=self.weights)
        # proposal_indices = np.random.choice(self.particle_indices, self.MAX_VIZ_PARTICLES)
        self.publish_particles(self.particles[proposal_indices,:])
      else:
        self.publish_particles(self.particles)

  def publish_particles(self, particles):
    pa = PoseArray()
    pa.header = Utils.make_header("map")
    pa.poses = Utils.particles_to_poses(particles)
    self.particle_pub.publish(pa)

if __name__ == '__main__':
  rospy.init_node("particle_filter", anonymous=True)
  pf = ParticleFilter()
  rospy.spin()

