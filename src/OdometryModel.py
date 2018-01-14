#!/usr/bin/env python

import numpy as np

class OdometryModel:

  def __init__(self):
    #self.last_pose = None
    pass
  
  def apply_motion_model(self, proposal_dist, action):
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

    v, delta, dt = action
    beta = np.arctan(0.5 * np.tan(delta))
    sin2beta = np.sin(2 * beta)
    dx = v * np.cos(proposal_dist[:, 2]) * dt
    dy = v * np.sin(proposal_dist[:, 2]) * dt
    #dtheta = ((v / self.CAR_LENGTH) * sin2beta) * dt  # Scale by dt
    dtheta = v*(np.tan(delta) / 0.25)* dt
    proposal_dist[:, 0] += dx + np.random.normal(loc=0.0, scale=0.05, size=4000)
    proposal_dist[:, 1] += dy + np.random.normal(loc=0.0, scale=0.025, size=4000)
    proposal_dist[:, 2] += dtheta + np.random.normal(loc=0.0, scale=0.25, size=4000)
