# Self-Learning-Cars-2D
A toy environment for reinforcement learning. In the single-agent version, a car is given the task of driving to a randomly assigned target. In the multi-agent version, one car chases and the other car tries to escape.
This is a project I started as a fun excercise to improve my engineering skills in deep reinforcement learning.
The task is not complex, but it's not a given that the agent learns to solve it, which makes it an interesting challenge. 
The main goals where to explore the effects on agent behaviour and performance of the following features:
- different reward functions (sparse, shaped, non-linear etc.)
- different observation spaces
- different action spaces


In brief, this repository provides two things:  
- a reinforcement learning environment (implementing the common OpenAI Gym standard)
- scripts for executing training with PPO (HER + SAC in the works)
### The Gym Environment

### The Training Scripts
The scripts that start with "MAIN" are the ones that have to be executed in order to start the training process. They set up or load the agents (imported from stable-baselines), initializing the environment, starting and logging the training, saving & loading the trained models and the environment parameters.  
#### Multi Agent/Attacker and Escaper
In the case of "multi-agent" training, two agents take turns in training, since stable-baselines does not support true multi-agent training (where multiple agents learn from the same episodes). The way I set this up here is by initializing two environments, one for the attacker and one for the escaper. The constructor for the environment gets passed an argument that determines the role of the agent. ```isEscaping=True``` would mean the agent being trained is the escaper. The environment constructor also requires an enemy model, so the agent can actually train against an adversery. It would seem like there might be an infinite regress problem here, because an agent needs an adversery to start training but the adversary needs and adversary to need training and so on - but because an untrained model is a valid model that already has a policy (even though a random one), training can be started without issue.
<img src="Pictures/architecture.svg"/>


#### Action space
```MultiDiscrete```  
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
