from random import seed
import gym

from stable_baselines.common.policies import MlpLstmPolicy, MlpPolicy, ActorCriticPolicy
from stable_baselines.sac.policies import MlpPolicy as SacMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2, SAC, HER
from stable_baselines.common.schedules import ConstantSchedule, LinearSchedule

from ENV_car_gym_environment import CustomEnv
from my_dynamic_learning_rate import ExpLearningRate
import car_utils

timesteps = 100000
attacker_model_folder = "../Models/"
attacker_tensorboard_folder = "../TensorboardLogs/2021_CARSTRIAL"
algorithm = "PPO2"
attacker_seed = 1

#my_learning_rate = LinearSchedule(timesteps, 0.005, 0.0001).value  # default: 0.00025
my_learning_rate = 0.0005
lr_string = "linearSchedule_"
dynamicLR = False
if(dynamicLR):
    # for dynamic LRs:
    lr_start = 0.0005
    lr_end = 0.00025
    #lr_end = 0.000063
    half_life = 0.5
    dyn_lr = ExpLearningRate(
        timesteps=timesteps, lr_start=lr_start, lr_min=lr_end, half_life=half_life)
    my_learning_rate = dyn_lr.value
    lr_string = "dynLR" + str(lr_start) + "-" + str(lr_end) + "_"


attacker_step_limit = 120
attacker_step_size = 0.01745*11.25
attacker_maxspeed = 5
attacker_acceleration = attacker_maxspeed/4
attacker_random_pos = True
attacker_binary_reward = True
attacker_rotation_vector = True
attacker_flat_obs_space = True
attacker_discrete_actionspace = True
custom_neural_net = "None"

attacker_params = car_utils.Env_Params(attacker_step_limit, attacker_step_size,
                                       attacker_maxspeed, attacker_acceleration, attacker_random_pos, attacker_binary_reward, attacker_rotation_vector, attacker_discrete_actionspace)


print("CARS_PPO2_DISCRETE.py LESS GO")

preview_env = CustomEnv(step_limit=attacker_step_limit, step_size=attacker_step_size, maxspeed=attacker_maxspeed, acceleration=attacker_acceleration, random_pos=attacker_random_pos,
                        binary_reward=attacker_binary_reward, rotation_vector=attacker_rotation_vector, discrete_actionspace=attacker_discrete_actionspace, flat_obs_space=attacker_flat_obs_space)

#env = CustomEnv(step_limit=my_step_limit, step_size = my_step_size, maxspeed = my_maxspeed,acceleration=my_acceleration, randomBall = my_randomBall, binaryReward= my_binaryReward) # 0.01745*5
if(algorithm=="PPO2"):
    attacker_env = make_vec_env(CustomEnv, n_envs=16, env_kwargs={'step_limit':attacker_step_limit, 'step_size' : attacker_step_size, 'maxspeed' : attacker_maxspeed, 'acceleration' : attacker_acceleration, 'random_pos' : attacker_random_pos, 'binary_reward' : attacker_binary_reward, 'rotation_vector' : attacker_rotation_vector, 'flat_obs_space' : attacker_flat_obs_space, 'discrete_actionspace' : attacker_discrete_actionspace})
elif(algorithm=="SAC" or algorithm =="HER"):
    attacker_env = CustomEnv(step_limit=attacker_step_limit, step_size = attacker_step_size, maxspeed = attacker_maxspeed,acceleration=attacker_acceleration, random_pos= attacker_random_pos, binary_reward= attacker_binary_reward, rotation_vector= attacker_rotation_vector, discrete_actionspace=attacker_discrete_actionspace, flat_obs_space= attacker_flat_obs_space) # 0.01745*5
    if(attacker_discrete_actionspace):
        print("ERROR: You used SAC with a discrete action space environment, but SAC requires a box action space!")
    #vectorized_attacker_env = DummyVecEnv([lambda: attacker_env])
else:
    print("ERROR: The algorithm you entered is not implemented: " + algorithm)

# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])


name = "2021_" + algorithm + "_CARS_LR_"  + lr_string + "timesteps_" + str(timesteps) + "seed_" + str(attacker_seed)  + attacker_params.toString()

# CRAZYDEEP6:
p_quarks = dict(net_arch=[dict(
    vf=[1024, 1024, 1024, 1024, 1024], pi=[256, 256, 128])])

if(algorithm=="SAC"):
    model = SAC(SacMlpPolicy, attacker_env, learning_rate=my_learning_rate,
                verbose=1, tensorboard_log=attacker_tensorboard_folder, seed = attacker_seed)
elif(algorithm == "PPO2"):
    if(custom_neural_net is not "None"):
        model = PPO2(MlpPolicy, attacker_env,policy_kwargs= p_quarks ,nminibatches=16, learning_rate=my_learning_rate,
                     verbose=1, tensorboard_log=attacker_tensorboard_folder, seed=attacker_seed)
    else:
        model = PPO2(MlpPolicy, attacker_env, nminibatches=16, learning_rate=my_learning_rate,
                    verbose=1, tensorboard_log=attacker_tensorboard_folder, seed=attacker_seed)
elif(algorithm=="HER"):
    model = HER('MlpPolicy', attacker_env, model_class=SAC,
                         random_exploration=0.1, verbose=1, tensorboard_log=attacker_tensorboard_folder)


model.learn(total_timesteps=timesteps, tb_log_name= name)



model.save(attacker_model_folder + name)


car_utils.saveEnvParameters(name, attacker_params)

while True:
    car_utils.showPreview(preview_env, model, attacker_step_limit)
