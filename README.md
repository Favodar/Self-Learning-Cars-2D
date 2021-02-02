# Self-Learning-Cars-2D
A toy environment for reinforcement learning. In the single-agent version, a car is given the task of driving to a randomly assigned target. In the multi-agent version, one car chases and the other car tries to escape.
This is a project I started as a fun excercise to improve my engineering skills in deep reinforcement learning.
The task is not complex, but it's not a given that the agent learns to solve it, which makes it an interesting challenge. 
The main goals where to explore the effects on agent behaviour and performance of the following features:
- different reward functions (sparse, shaped, non-linear etc.)
- different observation spaces
- different action spaces


In brief this repository provides two things:  
- a reinforcement learning environment (implementing the common OpenAI Gym standard)
- scripts for executing training, meaning: setting up the rl-algorithm (imported from stable-baselines), initializing the environment, starting and logging the training, saving & loading the trained models and the environment parameters
#### The Gym Environment

#### The Training Scripts

<img src="Pictures/architecture.svg"/>


#### Action space
```[a1, a2]```  
  
a1 = [0/1/2] = throttle [no/half/full]  
a2 = [0/1/2] = steer [left/straight/right]  
all action values are integers.

#### Observation space
```[[x1, y1], [x2, y2], [s2, r2]]```  
  
[x1, y1] = coordinates enemy in 2D euclidian space  
[x2, y2] = coordinates self in 2D euclidian space  
[s2, r2] = self speed, self rotation  
All observed values are floating point numbers.
