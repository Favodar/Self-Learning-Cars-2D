from env_params_class import Env_Params
def saveEnvParameters(filename, envparams: Env_Params):
    paramlist = [envparams.step_limit, envparams.step_size, envparams.maxspeed,
                          envparams.acceleration, envparams.random_pos, envparams.binary_reward, envparams.rotation_vector, envparams.discrete_actionspace]
    try:
        f = open("../Envparameters/envparameters_" + filename, "x")
        f.write(str(paramlist))
        f.close()
        print("saved envparameters to file.")
    except:
        print("envparameters couldn't be saved. They are:" +
            str(paramlist))


def loadEnvParameters(filename) -> Env_Params:

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

def showPreview(env, model, step_limit, episodes = 5, fps = 100):
    for i in range(episodes):
        obs = env.reset()
        for i in range(step_limit):
            action, _states = model.predict(obs)
            print(action)
            obs, rewards, dones, info = env.step(action)
            env.renderSlow(fps)
            if(dones):
                env.renderSlow(1)
                break





