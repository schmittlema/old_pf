#!/usr/bin/env python

import numpy as np

class OdometryModel:

  def __init__(self):
    #self.last_pose = None
    pass
  '''
  def motion_cb(self, msg)
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
  '''  
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
    # rotate the control into the coordinate space of each particle

    cosines = np.cos(proposal_dist[:,2])
    sines = np.sin(proposal_dist[:,2])

    proposal_dist[:,0] += cosines*control[0] - sines*control[1]
    proposal_dist[:,1] += sines*control[0] + cosines*control[1]
    proposal_dist[:,2] += control[2]

    add_rand = 0.01
    proposal_dist[:,0] += np.random.normal(loc=0.0,scale=add_rand,size=proposal_dist.shape[0])
    proposal_dist[:,1] += np.random.normal(loc=0.0,scale=add_rand*0.5,size=proposal_dist.shape[0])
    proposal_dist[:,2] += np.random.normal(loc=0.0,scale=0.25,size=proposal_dist.shape[0])
