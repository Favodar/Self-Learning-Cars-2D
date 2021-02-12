import gym
from numpy.lib import utils
from numpy.lib.npyio import save
from stable_baselines.common import save_util

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.schedules import ConstantSchedule, LinearSchedule

from ENV_car_gym_environment import CustomEnv
from ENV_multicar_gym_environment import CustomEnv as MultiEnv
import car_utils
from my_dynamic_learning_rate import ExpLearningRate

"""
This script loads pretrained attacker and defender models and lets them take turns in training against the most recent enemy model
"""

print("Load_and_train_CARS_MULTI_PPO2.py LESS GO")

# Name of pretrained model. For your convenience I've dropped pretrained attacker and defender models with a few million steps of training in the "Models" folder so you don't need to start from scratch ;-)
attacker_filename = "FURTHERCARS_NIGHT_ATTACKER_ppo2_LR_LinearSchedule_timesteps_100000ep_length_500turnrate_0.1963125maxspeed_2.5randomBall_TruebinaryReward_True"
attacker_model_iteration = "23070"
# Load environment parameters from file:
attacker_params = car_utils.loadEnvParameters(attacker_filename)

# First, initialize single-agent environment with parameters:
env = CustomEnv(step_limit=attacker_params.step_limit, step_size=attacker_params.step_size, maxspeed=attacker_params.maxspeed,
                acceleration=attacker_params.acceleration, random_pos=attacker_params.random_pos, binary_reward=attacker_params.binary_reward)  # 0.01745*5

# Load trained model:

attacker_model_folder = "../Models/"
attacker_model = PPO2.load(attacker_model_folder +
                           attacker_filename + attacker_model_iteration)
# Keep a copy of the loaded model for comparison purposes during the training process:
attacker_original_model = PPO2.load(attacker_model_folder +
                                    attacker_filename + attacker_model_iteration)

# Display a few example episodes with the loaded attacker model
showAttackerPreview = False
if(showAttackerPreview):
    car_utils.showPreview(env, attacker_model, attacker_params.step_limit)

defender_filename = "FURTHERCARS_NIGHT_ESCAPER_ppo2_LR_LinearSchedule_timesteps_100000ep_length_500turnrate_0.1963125maxspeed_2.25randomBall_TruebinaryReward_True"
defender_model_iteration = "23070"

defender_params = car_utils.loadEnvParameters(defender_filename)

defender_model_folder = "../Models/"
defender_model = PPO2.load( defender_model_folder + defender_filename + defender_model_iteration)
# Keep a copy of the loaded model for later comparison purposes
defender_original_model = PPO2.load(defender_model_folder + defender_filename + defender_model_iteration)


customParameters = True

if(customParameters):
    print("Training with custom environment parameters, which might differ from the env params the loaded model has been trained on.")
    print("This might cause bad agent performance and/or unexpected agent behaviour.")
    attacker_params.step_limit = 500
    defender_params.step_limit = attacker_params.step_limit
    defender_params.step_size = attacker_params.step_size*2
    defender_params.maxspeed = attacker_params.maxspeed*0.9
    defender_params.acceleration = attacker_params.acceleration*0.9*2
    attacker_params.acceleration = attacker_params.acceleration/2




# Custom names for the to-be-trained models
defender_name = "DEFENDER_ExpLR2asdfs_" + "ep_length_" + str(defender_params.step_limit) + "turnrate_" + str(
    defender_params.step_size) + "maxspeed_" + str(defender_params.maxspeed) + "randomBall_" + str(defender_params.random_pos) + "binaryReward_" + str(defender_params.binary_reward)
attacker_name = "ATTACKER_ExpLR2asdfs_" +  "ep_length_" + str(attacker_params.step_limit) + "turnrate_" + str(
    attacker_params.step_size) + "maxspeed_" + str(attacker_params.maxspeed) + "randomBall_" + str(attacker_params.random_pos) + "binaryReward_" + str(attacker_params.binary_reward)

attacker_tensorboard_folder = "../TensorboardLogs/reset-num-false"
defender_tensorboard_folder = "../TensorboardLogs/reset-num-false"   

car_utils.saveEnvParameters(attacker_name, attacker_params)
car_utils.saveEnvParameters(defender_name, defender_params)


# Initialize multi-agent environments with environment parameters and enemy models:
defender_env = MultiEnv(random_pos=defender_params.random_pos,step_limit=defender_params.step_limit, step_size = defender_params.step_size, maxspeed = defender_params.maxspeed,acceleration=defender_params.acceleration, binary_reward= defender_params.binary_reward, isEscaping= True, enemy_model = attacker_model, enemy_step_limit= attacker_params.step_limit, enemy_step_size= attacker_params.step_size, enemy_maxspeed= attacker_params.maxspeed, enemy_acceleration= attacker_params.acceleration)
attacker_env = MultiEnv(random_pos=attacker_params.random_pos,step_limit=attacker_params.step_limit, step_size = attacker_params.step_size, maxspeed = attacker_params.maxspeed,acceleration=attacker_params.acceleration, binary_reward= attacker_params.binary_reward, isEscaping= False, enemy_model = defender_model, enemy_step_limit= defender_params.step_limit, enemy_step_size= defender_params.step_size, enemy_maxspeed= defender_params.maxspeed, enemy_acceleration= defender_params.acceleration)

vectorized_env = DummyVecEnv([lambda: attacker_env])
vectorized_env2 = DummyVecEnv([lambda: defender_env])

attacker_model.set_env(vectorized_env)
defender_model.set_env(vectorized_env2)

attacker_model.tensorboard_log = attacker_tensorboard_folder
defender_model.tensorboard_log = defender_tensorboard_folder

timesteps_per_turn = 200000
# scheduler = LinearSchedule(timesteps, 0.001, 0.0001)
my_learning_rate = 0.0005 #scheduler.value

dynamicLR = True
if(dynamicLR):
    # for dynamic LRs:
    timesteps = 1000000
    lr_start = 0.0005
    lr_end = 0.000063
    half_life = 0.2
    dyn_lr = ExpLearningRate(
        timesteps=timesteps, lr_start=lr_start, lr_min=lr_end, half_life=half_life, save_interval=timesteps_per_turn)
    my_learning_rate = dyn_lr.value

attacker_model.learning_rate = my_learning_rate
defender_model.learning_rate = my_learning_rate 

showPreview = False



# The training process
for i in range(1, 10000):
    # Defender and attacker take turns in training (stable baselines does not support multiagent training)
    attacker_model.learn(total_timesteps= timesteps_per_turn, tb_log_name= attacker_name + str(i), log_interval=100)
    defender_model.learn(total_timesteps=timesteps_per_turn, tb_log_name= defender_name + str(i), log_interval=100)
    
    
    if(dynamicLR):
        dyn_lr.count()

    if(i%2==0):
        attacker_model.save(
            "../Models/" + attacker_name+str(i*timesteps_per_turn))
        defender_model.save(
            "../Models/" + defender_name+str(i*timesteps_per_turn))
        
        # If preview is turned on, every so often the agents demonstrate what they have learned:
        if(showPreview):
            car_utils.showPreview(attacker_env, attacker_model, attacker_params.step_limit)




