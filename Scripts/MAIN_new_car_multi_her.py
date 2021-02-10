import gym
from numpy.lib.npyio import save
from stable_baselines.common import save_util

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import HER
from stable_baselines import SAC
from stable_baselines.common.schedules import ConstantSchedule, LinearSchedule

from ENV_car_gym_environment import CustomEnv
from ENV_multicar_gym_environment import CustomEnv as MultiEnv
from ENV_multicar_gym_environment_goalenv import MultiCarGoalEnv as GoalEnv
import car_utils
from my_dynamic_learning_rate import ExpLearningRate

timesteps_per_turn = 1000000

defender_model_folder = "../Models/"
attacker_model_folder = "../Models/"
attacker_tensorboard_folder = "../TensorboardLogs/HER-test"
defender_tensorboard_folder = "../TensorboardLogs/HER-test"

# scheduler = LinearSchedule(timesteps_per_turn, 0.001, 0.0001)
# my_learning_rate = scheduler.value # 0.0005 

# for dynamic LRs:
timesteps = 1000000
lr_start = 0.001
lr_end = 0.000063
half_life = 0.1
dyn_lr = ExpLearningRate(
    timesteps=timesteps, lr_start=lr_start, lr_min=lr_end, half_life=half_life, save_interval=timesteps_per_turn)
my_learning_rate = dyn_lr.value # 0.0005  # 0.000063 #scheduler.value # 0.0005 default: 2.5e-4=0.00025
#scheduler = LinearSchedule(schedule_timesteps= timesteps,initial_p= lr_start, final_p = lr_end)

#print_LR = str(my_learning_rate) 
print_LR = str(lr_start) + "-" + str(lr_end)

attacker_step_limit = 200
attacker_turnrate = 0.01745*(11.25/2)
attacker_maxspeed = 2.5
attacker_acceleration = 2.5/2
attacker_random_pos = True
attacker_binaryReward = True

attacker_params = car_utils.Env_Params(attacker_step_limit, attacker_turnrate, attacker_maxspeed, attacker_acceleration, attacker_random_pos, attacker_binaryReward)

# Initialize stub environment to break vicious circle
stub_env = GoalEnv(discrete_actionspace=False,random_pos= attacker_random_pos, step_limit=attacker_step_limit, step_size = 0, maxspeed = 0,acceleration=0, binary_reward= attacker_binaryReward, isEscaping= False, enemy_model = 0, enemy_step_limit= 0, enemy_step_size= 0, enemy_maxspeed= 0, enemy_acceleration= 0)

model_class = SAC
random_exploration = 0.1
attacker_model = HER('MlpPolicy', stub_env, model_class=model_class,random_exploration = random_exploration, verbose=1, tensorboard_log=attacker_tensorboard_folder)
#attacker_model = PPO2(MlpPolicy, stub_env, learning_rate= my_learning_rate, verbose=1, tensorboard_log=attacker_tensorboard_folder)



print("MAIN_new_car_multi_her.py")

defender_step_limit = attacker_step_limit
defender_turnrate = attacker_turnrate*2
defender_maxspeed = attacker_maxspeed*0.8
defender_acceleration = attacker_acceleration*0.8
defender_random_pos = True
defender_binary_reward = True

defender_params = car_utils.Env_Params(defender_step_limit, defender_turnrate, defender_maxspeed, defender_acceleration, defender_random_pos, defender_binary_reward)


defender_env = MultiEnv(random_pos=defender_params.random_pos,step_limit=defender_params.step_limit, step_size = defender_params.step_size, maxspeed = defender_params.maxspeed,acceleration=defender_params.acceleration, binary_reward= defender_params.binary_reward, isEscaping= True, enemy_model = attacker_model, enemy_step_limit= attacker_params.step_limit, enemy_step_size= attacker_params.step_size, enemy_maxspeed= attacker_params.maxspeed, enemy_acceleration= attacker_params.acceleration)

# Custom names for the to-be-trained models
defender_name = "DEF_htest_rdmPos_PPO2_DEFENDER_" + "ep_length_" + str(defender_params.step_limit) + "turnrate_" + str(
    defender_params.step_size) + "maxspeed_" + str(defender_params.maxspeed) + "randomPos_" + str(defender_params.random_pos) + "binaryReward_" + str(defender_params.binary_reward)
attacker_name = "ATK_htest_rotVec_rdmPos_HERSAC_ATTACKER_" +  "ep_length_" + str(attacker_params.step_limit) + "turnrate_" + str(
    attacker_params.step_size) + "maxspeed_" + str(attacker_params.maxspeed) + "randomPos_" + str(attacker_params.random_pos) + "binaryReward_" + str(attacker_params.binary_reward)
 
defender_model = PPO2(MlpPolicy, defender_env, learning_rate= my_learning_rate, verbose=1, tensorboard_log=defender_tensorboard_folder)

attacker_env = GoalEnv(static_enemy=True,discrete_actionspace=False,random_pos=attacker_params.random_pos,step_limit=attacker_params.step_limit, step_size = attacker_params.step_size, maxspeed = attacker_params.maxspeed,acceleration=attacker_params.acceleration, binary_reward= attacker_params.binary_reward, isEscaping= False, enemy_model = defender_model, enemy_step_limit= defender_params.step_limit, enemy_step_size= defender_params.step_size, enemy_maxspeed= defender_params.maxspeed, enemy_acceleration= defender_params.acceleration)
vectorized_attacker_env = DummyVecEnv([lambda: attacker_env])
attacker_model.set_env(vectorized_attacker_env)

car_utils.save_env_parameters(attacker_name, attacker_params)
car_utils.save_env_parameters(defender_name, defender_params)

# Init learning
attacker_model.learn(total_timesteps=200000, tb_log_name= attacker_name + "INIT")
attacker_model.learn(total_timesteps=100, tb_log_name= attacker_name + "INIT2", reset_num_timesteps= True)

showPreview = True

for i in range(1, 10000):
    defender_model.learn(total_timesteps=timesteps_per_turn, tb_log_name= defender_name, log_interval=100, reset_num_timesteps=False)
    attacker_model.learn(total_timesteps= timesteps_per_turn//10, tb_log_name= attacker_name, log_interval=100, reset_num_timesteps=False)
    dyn_lr.count()

    if(i%10==0):
        attacker_model.save(attacker_model_folder + attacker_name+str(i*timesteps_per_turn))
        defender_model.save(defender_model_folder + defender_name+str(i*timesteps_per_turn))
        
        # If preview is turned on, every so often the agents demonstrate what they have learned:
        if(showPreview):
            for i in range(5):
                obs = attacker_env.reset()
                for i in range(attacker_params.step_limit):
                    action, _states = attacker_model.predict(obs)
                    print(action)
                    obs, rewards, dones, info = attacker_env.step(action)
                    attacker_env.renderSlow(200)
                    if(dones):
                        attacker_env.renderSlow(1)
                        break
    

while True:
    obs = defender_env.reset()
    for i in range(attacker_step_limit):
        action, _states = defender_model.predict(obs)
        obs, rewards, dones, info = defender_env.step(action)
        defender_env.renderSlow(25)
