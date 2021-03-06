import math
import random
import time

import gym
import numpy as np
from gym import spaces

from car_physics import Ball, Car, Render


class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  step_counter = 0
  episode_counter = 0
  
  def __init__(self, step_limit, step_size, maxspeed, acceleration, random_pos = False, binary_reward = False, rotation_vector = False, flat_obs_space = False, discrete_actionspace = True):

    self.step_limit = step_limit
    self.step_size = step_size
    self.maxspeed = maxspeed
    self.acceleration = acceleration
    self.random_pos = random_pos
    self.binary_reward = binary_reward
    self.rotation_vector = rotation_vector
    self.flat_obs_space = flat_obs_space

    self.number_of_actions = 2
    self.number_of_cars = 1
    self.episodeIsOver = False

    self.myCar = Car(rotation_step_size = step_size, maxspeed= maxspeed, acceleration=acceleration)
    if(random_pos):
      self.spawnBall()
    else:
      self.myBall = Ball(75, 75)

    self.myRender = Render([self.myCar, self.myBall])

    # super(CustomEnv, self).__init__()

    actionlist = [3 for j in range(self.number_of_actions)]
    if(discrete_actionspace):
        self.action_space = spaces.MultiDiscrete(actionlist)
    else:
        self.action_space = spaces.Box(
            np.array([-1, 0]), np.array([2, 2]), dtype=np.float32)

    self.observation = self.getObservation()

    if(flat_obs_space):
      self.observation_space = spaces.Box(-np.inf, np.inf, shape=np.array(self.observation).shape, dtype=np.float32) #self.number_of_cars,
    else:
      self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,2), dtype=np.float32) #self.number_of_cars,
      



  def step(self, action):

    assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

    self.myCar.move(action[0], action[1])

    #self.renderSlow(400)



    observation = self.getObservation()
    
    #print(observation)
    reward = self.getReward()

    self.step_counter += 1

    #if(self.episode_counter % 100 == 0):
    #    self.renderSlow(400)

    done = (self.episodeIsOver|(self.step_counter>=self.step_limit))

    # info = "I don't know what 'info' is supposed to contain."

    return observation, reward, done, {} # info

  def getReward(self):

        coordinates1 = self.myCar.coordinates[0]
        coordinates2 = self.myBall.coordinates[0]

        xdistance = coordinates1[0] - coordinates2[0]
        ydistance = coordinates1[1] - coordinates2[1]                
        distance = math.sqrt(xdistance**2 + ydistance**2)

        #print ("distance = " + str(distance))

        if(self.binary_reward):
          if(distance<5):
            self.episodeIsOver = True
            return 1
          else:
            return 0
        
        if(distance<0.01):
            distance = 0.01

        reward = 10/(distance/10)+(35-distance)#abs(35-distance)*(35-distance)

        #print ("reward = " + str(reward))

        return reward

  def reset(self):
    self.render()

    self.step_counter = 0
    self.episodeIsOver = False
    self.episode_counter += 1

    self.myCar = Car(rotation_step_size = self.step_size, maxspeed= self.maxspeed, acceleration=self.acceleration)
    if(self.random_pos):
      self.spawnBall()

    self.myRender.setObjects([self.myCar, self.myBall])

    observation = self.getObservation()
    #observation = [self.myCar.coordinates[0][0], self.myCar.coordinates[0][1]]#, self.myCar.speed, self.myCar.rotation]


    return observation  # reward, done, info can't be included

  def render(self, mode='human'):
    self.myRender.renderFrame(self.getReward(), self.episode_counter)

  def renderSlow(self, fps):
    self.myRender.renderFrame(self.getReward(), self.episode_counter)
    time.sleep(1.0/fps)



  def close(self):
        print ("close() has been called")


  def spawnBall(self):

        sign1 = random.choice([1, -1])
        sign2 = random.choice([1, -1])
        x = 50 + sign1*random.randint(40, 50)
        y = 50 + sign2*random.randint(40, 50)
        
        
        self.myBall = Ball(x, y)

  def getObservation(self):

    if not (self.rotation_vector):
      if(self.flat_obs_space):
        return [self.myBall.coordinates[0][0], self.myBall.coordinates[0][1], self.myCar.coordinates[0][0], self.myCar.coordinates[0][1], self.myCar.speed, self.myCar.rotation]
      else:
        return [[self.myBall.coordinates[0][0], self.myBall.coordinates[0][1]],[self.myCar.coordinates[0][0], self.myCar.coordinates[0][1]],[self.myCar.speed, self.myCar.rotation]]
    else:
      return [self.myBall.coordinates[0][0], self.myBall.coordinates[0][1], self.myCar.coordinates[0][0], self.myCar.coordinates[0][1], self.myCar.speed, self.myCar.rotation_vector_x, self.myCar.rotation_vector_y]
