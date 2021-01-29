import gym
from numpy.lib.npyio import save
from stable_baselines.common import save_util

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.schedules import ConstantSchedule, LinearSchedule

from car_gym_environment import CustomEnv
from multicar_gym_environment import CustomEnv as MultiEnv
import car_utils


#filename = "CARS_medium5_225_newObs_ppo2_LR_LinearSchedule_timesteps_4000000ep_length_120turnrate_0.1963125maxspeed_2.5randomBall_TruebinaryReward_True40"# "CARS_medium5_225_newObs_ppo2_LR_LinearSchedule_timesteps_4000000ep_length_120turnrate_0.1963125maxspeed_2.5randomBall_TruebinaryReward_True"
#filename = "CARS_NIGHT_ATTACKER_ppo2_LR_LinearSchedule_timesteps_100000ep_length_500turnrate_0.1963125maxspeed_2.5randomBall_TruebinaryReward_True230"
attacker_filename = "FURTHERCARS_NIGHT_ATTACKER_ppo2_LR_LinearSchedule_timesteps_100000ep_length_500turnrate_0.1963125maxspeed_2.5randomBall_TruebinaryReward_True23070"
# Load signal parameters from file:
#f = open("../Envparameters/envparameters_" + filename, "r")
f = open("../Envparameters/envparameters_CARS_ATTACKER_ppo2_LR_LinearSchedule_timesteps_30000ep_length_300turnrate_0.1963125maxspeed_2.5randomBall_TruebinaryReward_True", "r")

attacker_params = car_utils.load_env_parameters(attacker_filename)

   

# Initialize environment with signal parameters:
env = CustomEnv(step_limit=attacker_params.step_limit, step_size=attacker_params.step_size, maxspeed=attacker_params.maxspeed,
                acceleration=attacker_params.acceleration, random_pos=attacker_params.random_pos, binary_reward=attacker_params.binary_reward)  # 0.01745*5

# Load trained model:
attacker_model_iteration = "23070"
attacker_model_folder = "/media/ryuga/Shared Storage/Models/"
attacker_model = PPO2.load(attacker_model_folder +
                           attacker_filename + attacker_model_iteration)
attacker_original_model = PPO2.load(attacker_model_folder +
                                    attacker_filename + attacker_model_iteration)


showAttackerPreview = True
if(showAttackerPreview):
    for i in range(5):
        obs = env.reset()
        for i in range(attacker_params.step_limit):
            action, _states = attacker_model.predict(obs)
            print(action)
            obs, rewards, dones, info = env.step(action)
            env.renderSlow(200)
            if(dones):
                env.renderSlow(1)
                break



print("Load_and_train_CARS_MULTI_PPO2.py LESS GO")

#filename2 = "CARS_NIGHT_ESCAPER_ppo2_LR_LinearSchedule_timesteps_100000ep_length_500turnrate_0.1963125maxspeed_2.25randomBall_TruebinaryReward_True"
defender_filename = "FURTHERCARS_NIGHT_ESCAPER_ppo2_LR_LinearSchedule_timesteps_100000ep_length_500turnrate_0.1963125maxspeed_2.25randomBall_TruebinaryReward_True"
defender_model_iteration = "23070"

# defender_params = car_utils.load_env_parameters(defender_filename)
defender_params = car_utils.load_env_parameters("CARS_ATTACKER_ppo2_LR_LinearSchedule_timesteps_30000ep_length_300turnrate_0.1963125maxspeed_2.5randomBall_TruebinaryReward_True")


defender_model_folder = "/media/ryuga/Shared Storage/Models/"
defender_model = PPO2.load( defender_model_folder + defender_filename + defender_model_iteration)
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


defender_name = "2021_DEFENDER_alternate_quickly_foxbunny_" + "ep_length_" + str(defender_params.step_limit) + "turnrate_" + str(
    defender_params.step_size) + "maxspeed_" + str(defender_params.maxspeed) + "randomBall_" + str(defender_params.random_pos) + "binaryReward_" + str(defender_params.binary_reward)
attacker_name = "2021_ATTACKER_alternate_quickly_foxbunny_" +  "ep_length_" + str(attacker_params.step_limit) + "turnrate_" + str(
    attacker_params.step_size) + "maxspeed_" + str(attacker_params.maxspeed) + "randomBall_" + str(attacker_params.random_pos) + "binaryReward_" + str(attacker_params.binary_reward)
   

car_utils.save_env_parameters(attacker_name, attacker_params)
car_utils.save_env_parameters(defender_name, defender_params)


# Initialize environment with environment parameters:
env2 = MultiEnv(random_pos=defender_params.random_pos,step_limit=defender_params.step_limit, step_size = defender_params.step_size, maxspeed = defender_params.maxspeed,acceleration=defender_params.acceleration, binary_reward= defender_params.binary_reward, isEscaping= True, enemy_model = attacker_model, enemy_step_limit= attacker_params.step_limit, enemy_step_size= attacker_params.step_size, enemy_maxspeed= attacker_params.maxspeed, enemy_acceleration= attacker_params.acceleration)
env = MultiEnv(random_pos=attacker_params.random_pos,step_limit=attacker_params.step_limit, step_size = attacker_params.step_size, maxspeed = attacker_params.maxspeed,acceleration=attacker_params.acceleration, binary_reward= attacker_params.binary_reward, isEscaping= False, enemy_model = defender_model, enemy_step_limit= defender_params.step_limit, enemy_step_size= defender_params.step_size, enemy_maxspeed= defender_params.maxspeed, enemy_acceleration= defender_params.acceleration)

vectorized_env = DummyVecEnv([lambda: env])
vectorized_env2 = DummyVecEnv([lambda: env2])

attacker_model.set_env(vectorized_env)
defender_model.set_env(vectorized_env2)

timesteps2 = 10000

scheduler = LinearSchedule(timesteps2, 0.001, 0.0001)
my_learning_rate2 = 0.0005 #scheduler.value

showPreview = True

for i in range(10000):
    # Defender and attacker take turns in training (stable baselines doesnt support multiagent training)
    defender_model.learn(total_timesteps=timesteps2, tb_log_name= defender_name + str(i), log_interval=100)
    attacker_model.learn(total_timesteps= timesteps2//2, tb_log_name= attacker_name + str(i), log_interval=100)

    if(i%2==0):
        attacker_model.save(
            "/media/ryuga/Shared Storage/Models/" + attacker_name+str(i))
        defender_model.save(
            "/media/ryuga/Shared Storage/Models/" + defender_name+str(i))
        
        # If preview is turned on, every so often the agents demonstrate what they have learned:
        if(showPreview):
            for i in range(5):
                obs = env.reset()
                for i in range(attacker_params.step_limit):
                    action, _states = attacker_model.predict(obs)
                    print(action)
                    obs, rewards, dones, info = env.step(action)
                    env.renderSlow(200)
                    if(dones):
                        env.renderSlow(1)
                        break




