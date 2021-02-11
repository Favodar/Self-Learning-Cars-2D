class Env_Params:
    def __init__(self, step_limit, step_size, maxspeed, acceleration, random_pos, binary_reward, discrete_actionspace = True):
        self.step_limit = step_limit
        self.step_size = step_size
        self.maxspeed = maxspeed
        self.acceleration = acceleration
        self.random_pos = random_pos
        self.binary_reward = binary_reward
        self.discrete_actionspace = discrete_actionspace
