#!/usr/bin/env python

import rospy
import numpy as np
import utils as Utils
from std_msgs.msg import Float64
from threading import Lock

class OdometryMotionModel:

  def __init__(self, particles, state_lock=None):
    self.last_pose = None # The last pose that was received
    self.particles = particles
    if state_lock is None:
      self.state_lock = Lock()
    else:
      self.state_lock = state_lock
    
  def motion_cb(self, msg):    
    # Compute the control from the msg and last_pose
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
      self.apply_motion_model(self.particles, control)

    self.last_pose = pose
    
  def apply_motion_model(self, proposal_dist, control):
    '''
    The motion model applies the odometry to the particle distribution. Since there the odometry
    data is inaccurate, the motion model mixes in gaussian noise to spread out the distribution.
    Vectorized motion model. Computing the motion model over all particles is thousands of times
    faster than doing it for each particle individually due to vectorization and reduction in
    function call overhead
    
    TODO this could be better, but it works for now
        - fixed random noise is not very realistic
        - ackermann model provides bad estimates at high speed
    '''
    # Update the proposal distribution by applying the control to each particle
    # rotate the control into the coordinate space of each particle

    cosines = np.cos(proposal_dist[:,2])
    sines = np.sin(proposal_dist[:,2])

    self.state_lock.acquire()
    proposal_dist[:,0] += cosines*control[0] - sines*control[1]
    proposal_dist[:,1] += sines*control[0] + cosines*control[1]
    proposal_dist[:,2] += control[2]

    add_rand = 0.01
    proposal_dist[:,0] += np.random.normal(loc=0.0,scale=add_rand,size=proposal_dist.shape[0])
    proposal_dist[:,1] += np.random.normal(loc=0.0,scale=add_rand*0.5,size=proposal_dist.shape[0])
    proposal_dist[:,2] += np.random.normal(loc=0.0,scale=0.25,size=proposal_dist.shape[0])    
    self.state_lock.release()

class KinematicMotionModel:

  def __init__(self, particles, state_lock=None):
    self.last_servo_cmd = None # The most recent servo command
    self.last_vesc_stamp = None # The time stamp from the previous vesc state msg
    self.particles = particles
    self.SPEED_TO_ERPM_OFFSET = float(rospy.get_param("/vesc/speed_to_erpm_offset")) # Offset conversion param from rpm to speed
    self.SPEED_TO_ERPM_GAIN   = float(rospy.get_param("/vesc/speed_to_erpm_gain"))   # Gain conversion param from rpm to speed
    self.STEERING_TO_SERVO_OFFSET = float(rospy.get_param("/vesc/steering_angle_to_servo_offset")) # Offset conversion param from servo position to steering angle
    self.STEERING_TO_SERVO_GAIN   = float(rospy.get_param("/vesc/steering_angle_to_servo_gain")) # Gain conversion param from servo position to steering angle
    
    if state_lock is None:
      self.state_lock = Lock()
    else:
      self.state_lock = state_lock
      
    # This subscriber just caches the most recent servo position command
    self.servo_pos_sub  = rospy.Subscriber(rospy.get_param("~servo_pos_topic", "/vesc/sensors/servo_position_command"), Float64,
                                       self.servo_cb, queue_size=1)
                                       

  def servo_cb(self, msg):
    self.last_servo_cmd = msg.data # Just update servo command

  def motion_cb(self, msg):
    
    if self.last_servo_cmd is None:
      return

    if self.last_vesc_stamp is None:
      print ("Vesc callback called for first time....")
      self.last_vesc_stamp = msg.header.stamp

    # Convert raw msgs to controls
    # Note that control = (raw_msg_val - offset_param) / gain_param
    curr_speed = (msg.state.speed - self.SPEED_TO_ERPM_OFFSET) / self.SPEED_TO_ERPM_GAIN
    curr_steering_angle = (self.last_servo_cmd - self.STEERING_TO_SERVO_OFFSET) / self.STEERING_TO_SERVO_GAIN
    dt = (msg.header.stamp - self.last_vesc_stamp).to_sec()
    self.apply_motion_model(proposal_dist=self.particles, control=[curr_speed, curr_steering_angle, dt])
    
    self.last_vesc_stamp = msg.header.stamp    
   
  def apply_motion_model(self, proposal_dist, control):
    
    # Update the proposal distribution by applying the control to each particle
    
    v, delta, dt = control
    beta = np.arctan(0.5 * np.tan(delta))
    sin2beta = np.sin(2 * beta)
    dx = v * np.cos(proposal_dist[:, 2]) * dt
    dy = v * np.sin(proposal_dist[:, 2]) * dt
    #dtheta = ((v / self.CAR_LENGTH) * sin2beta) * dt  # Scale by dt
    dtheta = v*(np.tan(delta) / 0.25)* dt
    
    self.state_lock.acquire()
    proposal_dist[:, 0] += dx + np.random.normal(loc=0.0, scale=0.05, size=proposal_dist.shape[0])
    proposal_dist[:, 1] += dy + np.random.normal(loc=0.0, scale=0.025, size=proposal_dist.shape[0])
    proposal_dist[:, 2] += dtheta + np.random.normal(loc=0.0, scale=0.25, size=proposal_dist.shape[0])
    self.state_lock.release()
