from stable_baselines.common.schedules import ConstantSchedule, LinearSchedule

class ExpLearningRate():

    """
    Logarithmically declining learning rate. This is a somewhat robust learning rate since it
    can apply learning rates of different orders of magnitude during one training, guaranteeing
    to hit the optimal static learning rate at some point.
    If you know upper and lower bounds of a sensible learning rate (those are dependent on
    environment and algorithm!), plug them in and adapt the half life accordingly.
    Default values are a compromise between robustness and efficiency.
    :param timesteps: (int) This is the number used for half-life calculation. It is recommended to put in the number of steps you plan to train your agent for, though any number can be chosen here.
    :param lr_start: (float) the starting value of the learning rate.
    :param lr_min: (float) clips the learning rate to a minimum value. Set to 0 for no clipping.
    :param half_life: (float) The (relative) fraction of timesteps after which the learning rate will reach half its initial value. E.g. 0.1 = 10%, Meaning half_life is 100,000 steps when training for 1,000,000 steps
    """
    
    def __init__(self, timesteps, lr_start = 0.001, lr_min = 0.00001, half_life = 0.1, save_interval = 0):
        self.timesteps = timesteps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.half_life_factor = 1/half_life
        self.save_interval = save_interval
        self.counter = 1

    def value(self, step):
        print("step:" + str(step))
        if(self.save_interval==0):
            s = (self.timesteps-(step*self.timesteps))
        else:
            s = (self.save_interval*self.counter-(step*self.save_interval))
        lr = self.lr_start*0.5**(s*(self.half_life_factor/self.timesteps))
        #print("LR: " + str(lr))
        return max(lr, self.lr_min)

    def count(self):
        self.counter += 1

class LearningRate():
    
    dyn_lr = None
    
    def __init__(self, lr_function, timesteps, save_interval, lr_start, lr_min=0, half_life=0.5, use_recommended_hyperparams = False):
        
        self.lr_string = lr_function + "_"
        if(use_recommended_hyperparams):
            lr_start = 0.0005
            lr_min = 0.00025 #lr_end = 0.000063
            half_life = 0.5
        
        if(lr_function=="linear"):
            my_learning_rate = LinearSchedule(timesteps, lr_start, lr_min).value  # default: 0.00025
            if(save_interval!=0):
                print("Warning: linear learning rates are possibly broken when training is paused and resumed (i.e. if preview is enabled)")

        elif(lr_function=="exponential"):
            # for exponentially decaying learning rates:


            self.dyn_lr = ExpLearningRate(
                timesteps=timesteps, lr_start=lr_start, lr_min=lr_min, half_life=half_life, save_interval=save_interval) # add save_interval parameter when preview is on!
            self.my_learning_rate = self.dyn_lr.value
            self.lr_string += str(lr_start) + "-" + str(lr_min) + "_"
            if(save_interval!=0):
                print("Warning: When using decaying learning rates and training happens in intervals, the LearningRate.count() function must be called after each interval, or the decay will not work properly!")

        else:
            self.my_learning_rate = lr_start
            if(lr_function!="static"):
                print("No learning rate function of the name " + lr_function + " was found! Defaulting to static learning rate.")

    def count(self):
        if(self.dyn_lr!=None):
            self.dyn_lr.count()