from car_utils import Env_Params

def save_env_parameters(filename, envparams: Env_Params):
    paramlist = [envparams.step_limit, envparams.step_size, envparams.maxspeed,
                          envparams.acceleration, envparams.random_pos, envparams.binary_reward]
    try:
        f = open("../Envparameters/envparameters_" + filename, "x")
        f.write(str(paramlist))
        f.close()
        print("saved envparameters to file.")
    except:
        print("envparameters couldn't be saved. They are:" +
            str(paramlist))


def load_env_parameters(filename) -> Env_Params:

    # Load signal parameters from file:
    f = open("../Envparameters/envparameters_" + filename, "r")
    envparameters = f.read()
    envparameters = envparameters.strip('[')
    envparameters = envparameters.strip(']')
    f_list = [i for i in envparameters.split(",")]
    print("loaded envparameters from file: " + str(f_list))

    env_params_obj = Env_Params(int(f_list[0]),
                    float(f_list[1]),
                    float(f_list[2]),
                    float(f_list[3]),
                    bool(f_list[4]),
                    bool(f_list[5]))
    
    return env_params_obj



class Env_Params:
    def __init__(self, step_limit, step_size, maxspeed, acceleration, random_pos, binary_reward):
        self.step_limit = step_limit
        self.step_size = step_size
        self.maxspeed = maxspeed
        self.acceleration = acceleration
        self.random_pos = random_pos
        self.binary_reward = binary_reward
