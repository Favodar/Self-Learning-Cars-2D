import gym
from gym import spaces
import math
import time
import random
import numpy as np
from car_physics import Car, Render, Ball

class MultiCarGoalEnv(gym.GoalEnv):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  step_counter = 0
  episode_counter = 0
  
  def __init__(self, step_limit, step_size, maxspeed, acceleration, binary_reward, isEscaping, enemy_model, enemy_step_limit, enemy_step_size, enemy_maxspeed, enemy_acceleration, random_pos = False, discrete_actionspace = True, static_enemy = False):

    self.static_enemy = static_enemy
    self.enemy_model = enemy_model
    self.step_limit = step_limit
    self.step_size = step_size
    self.maxspeed = maxspeed
    self.acceleration = acceleration
    self.binaryReward = binary_reward
    self.isEscaping = isEscaping

    self.enemy_step_limit = enemy_step_limit
    self.enemy_step_size = enemy_step_size
    self.enemy_maxspeed = enemy_maxspeed
    self.enemy_acceleration = enemy_acceleration

    self.random_pos = random_pos

    self.number_of_actions = 2
    #self.number_of_cars = 1
    self.episodeIsOver = False

    pos1_x, pos1_y, pos1_rot = 50.0, 50.0, 0
    pos2_x, pos2_y, pos2_rot = 95.0, 95.0, math.pi*1.5

    if(isEscaping):
        self.my_x_pos = pos2_x
        self.my_y_pos = pos2_y
        self.my_rotation = pos2_rot
        self.enemy_x_pos = pos1_x
        self.enemy_y_pos = pos1_y
        self.enemy_rotation = pos1_rot
    else:
        self.my_x_pos = pos1_x
        self.my_y_pos = pos1_y
        self.my_rotation = pos1_rot
        self.enemy_x_pos = pos2_x
        self.enemy_y_pos = pos2_y
        self.enemy_rotation = pos2_rot

    self.resetCars()

    self.myRender = Render([self.myCar, self.enemy_car])

    # super(CustomEnv, self).__init__()

    actionlist = [3 for j in range(self.number_of_actions)]

    if(discrete_actionspace):
        self.action_space = spaces.MultiDiscrete(actionlist)
    else:
        self.action_space = spaces.Box(np.array([-1, 0]), np.array([2, 2]), dtype=np.float32)
    #self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,2), dtype=np.float32) #self.number_of_cars,
    obs = self.getObservation()
    self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['desired_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
    self.observation = obs





  def step(self, action):

    #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
    if not(self.static_enemy):
        enemy_action, _states = self.enemy_model.predict(self.getObservation(isEnemy=True))
        self.enemy_car.move(enemy_action[0], enemy_action[1])
    self.myCar.move(action[0], action[1])

    #self.renderSlow(50)
    self.render()

    self.observation = self.getObservation()

    #print(observation)
    # info = "I don't know what 'info' is supposed to contain."
    info = {}
    reward = self.compute_reward(self.observation['achieved_goal'], self.observation['desired_goal'], info)

    self.step_counter += 1

    #if(self.episode_counter%30==0):
    #    self.renderSlow(400)

    done = (self.episodeIsOver|(self.step_counter>=self.step_limit))

    

    return self.observation, reward, done, info # info



  def compute_reward(self, achieved_goal, goal, info):

    xdistance = achieved_goal[0] - goal[0]
    ydistance = achieved_goal[1] - goal[1]                
    distance = math.sqrt(xdistance**2 + ydistance**2)

    if(self.isEscaping):
        if(self.binaryReward):
            if(distance<10):
                self.episodeIsOver = True
                return -self.step_limit/2
            else:
                return 1

        if(distance<0.01):
            distance = 0.01

        reward = math.sqrt(distance)-10/((distance/10)**2) #10/(distance/10)+(35-distance)#abs(35-distance)*(35-distance)

        #print ("reward = " + str(reward))

        return reward


    else:
        #print ("distance = " + str(distance))

        if(self.binaryReward):
            if(distance<10):
                self.episodeIsOver = True
                return 1
            else:
                return 0

        if(distance<1):
            distance = 1

        reward = 30/(distance/10)-(10+math.sqrt(distance))#abs(35-distance)*(35-distance)

        #print ("reward = " + str(reward))

        return reward

  def reset(self):
    #self.render()
    print(str(self.observation))
    super(MultiCarGoalEnv, self).reset()

    self.step_counter = 0
    self.episodeIsOver = False
    self.episode_counter += 1

    self.resetCars()

    self.myRender.setObjects([self.myCar, self.enemy_car])

    self.observation = self.getObservation()

    


    return self.observation  # reward, done, info can't be included

  def render(self, mode='human'):
    # Display current state of environment, including current reward
    self.myRender.renderFrame(self.compute_reward(self.observation['achieved_goal'], self.observation['desired_goal'], {}))

  def renderSlow(self, fps):

    self.myRender.renderFrame(self.compute_reward(self.observation['achieved_goal'], self.observation['desired_goal'], {}))
    time.sleep(1.0/fps)



  def close(self):
        print ("close() has been called")


  def getRandomPosition(self):

        sign1 = random.choice([1, -1])
        sign2 = random.choice([1, -1])
        x = 50 + sign1*random.randint(40, 50)
        y = 50 + sign2*random.randint(40, 50)
        
        
        return [x, y]

  def getObservation(self, isEnemy = False):

    if(isEnemy):
        obs = np.array([[self.myCar.coordinates[0][0], self.myCar.coordinates[0][1]],[self.enemy_car.coordinates[0][0], self.enemy_car.coordinates[0][1]],[self.enemy_car.speed, self.enemy_car.rotation]])
        return obs
    else:
        obs = np.array([[self.enemy_car.coordinates[0][0], self.enemy_car.coordinates[0][1]],[self.myCar.coordinates[0][0], self.myCar.coordinates[0][1]],[self.myCar.speed, self.myCar.rotation_vector_x, self.myCar.rotation_vector_y]])
        
    return {
            'observation': np.concatenate(obs),
            'achieved_goal': self.myCar.coordinates[0],
            'desired_goal': self.enemy_car.coordinates[0]
        }

  def resetCars(self):
      if(self.random_pos):
        coordinates = self.getRandomPosition()
        if(self.isEscaping):
          self.my_x_pos = coordinates[0]
          self.my_y_pos = coordinates[1]
        else:
          self.enemy_x_pos = coordinates[0]
          self.enemy_y_pos = coordinates[1]
      self.myCar = Car(x_pos = self.my_x_pos, y_pos = self.my_y_pos, rotation=self.my_rotation ,rotation_step_size = self.step_size, maxspeed= self.maxspeed, acceleration=self.acceleration)
      self.enemy_car = Car(x_pos = self.enemy_x_pos,y_pos= self.enemy_y_pos, rotation= self.enemy_rotation, rotation_step_size= self.enemy_step_size, maxspeed= self.enemy_maxspeed, acceleration= self.enemy_acceleration)