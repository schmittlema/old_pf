import rospy
import numpy as np

class ReSampler:
  def __init__(self, particles, weights, state_lock=None):
    self.particles = particles
    self.weights = weights
    self.particle_indices = None  
    self.step_array = None
    
    if state_lock is None:
      self.state_lock = Lock()
    else:
      self.state_lock = state_lock
  
  def resample_naiive(self):
    # Use np.random.choice to re-sample 
    self.state_lock.acquire()
    if self.particle_indices is None:
      self.particle_indices = np.arange(self.particles.shape[0])
   
   
    proposal_indices = np.random.choice(self.particle_indices, self.particles.shape[0], p=self.weights)
   
    self.particles[:] = self.particles[proposal_indices,:]    
    self.weights[:] = 1.0 / self.particles.shape[0]
    
    self.state_lock.release()
  
  def resample_low_variance(self):
    self.state_lock.acquire()
    # Implement low variance re-sampling
    if self.step_array is None:
      self.step_array = (1.0/self.particles.shape[0]) * np.array(range(0,self.particles.shape[0]), dtype=np.float32)
    initval = np.random.uniform() * (1.0/self.particles.shape[0])
    vals    = initval + self.step_array
    cumwt   = np.cumsum(self.weights)
    proposal_indices = np.searchsorted(cumwt, vals, side='left')
    
    self.particles[:] = self.particles[proposal_indices,:] 
    self.weights[:] = 1.0 / self.particles.shape[0]
    
    self.state_lock.release()

