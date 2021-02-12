class Env_Params:
    def __init__(self, step_limit, step_size, maxspeed, acceleration, random_pos, binary_reward, rotation_vector=False, discrete_actionspace = True):
        self.step_limit = step_limit
        self.step_size = step_size
        self.maxspeed = maxspeed
        self.acceleration = acceleration
        self.random_pos = random_pos
        self.binary_reward = binary_reward
        self.rotation_vector = rotation_vector
        self.discrete_actionspace = discrete_actionspace

    def toString(self):
        return ("epLength_" + str(self.step_limit) + "_turnrate_" + str(
            self.step_size) + "_maxspeed_" + str(self.maxspeed) + "_randomPos_" + str(self.random_pos) + "_binaryReward_" + str(self.binary_reward)) + "_rotVec_" + str(self.rotation_vector) + "_discrete_" + str(self.discrete_actionspace)
