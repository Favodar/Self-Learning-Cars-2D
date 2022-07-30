import numpy as np
from abc import ABC, abstractmethod
import graphics
import time


class Object2D:

    @abstractmethod
    def get2DpointList(self):
        pass



class Car(Object2D):
    """
    A car that can move according to certain rules and initialization parameters.

    :param x_pos: (float) starting position x-coordinate
    :param y_pos: (float) starting position y-coordinate
    :param rotation: (float) starting rotation
    :param rotation_step_size: (float) rotation speed when steering
    :param border: (bool) If True, the cars can't go outside the screen and instead reappear on the other side of the screen when crossing a border. Else, the world is borderless and infinite (but cars outside the screen can't be observed)
    """
    def __init__ (self, x_pos = 50.0, y_pos = 50.0, rotation = 0, rotation_step_size = 0.01745*5, maxspeed = 1.0, border = False, acceleration = 0.1):
        self.rotation_step_size = rotation_step_size
        self.maxspeed = maxspeed
        self.border = border
        self.acceleration = acceleration

        self.rotation = rotation
        self.rotation_vector_x = np.sin(self.rotation)
        self.rotation_vector_y = np.cos(self.rotation)
        self.coordinates = np.array([[x_pos, y_pos],])
        self.speed = 0

        # print("Car initialized.")
        # print("rotation_step_size = " + str(self.rotation_step_size))
        # print("maxspeed = " + str(maxspeed))
        # print("joint coordinates:" +str(self.coordinates))
    
    def move(self, gaspedal=0, rotate=0):

        # If gaspedal is pushed accelerate, else deccelerate
        if(gaspedal > 0):
            self.speed += (self.acceleration*gaspedal)
        elif(self.speed > 0):
            self.speed -= (self.acceleration)

        # Make sure speed is not negative or above max speed or 0:
        if(self.speed > self.maxspeed):
            self.speed = self.maxspeed
        elif(self.speed < 0):
            self.speed = 0

        self.rotation += (rotate-1)*self.rotation_step_size

        self.rotation_vector_x = np.sin(self.rotation)
        self.rotation_vector_y = np.cos(self.rotation)

        self.coordinates[0][0] += self.speed*self.rotation_vector_x
        self.coordinates[0][1] += self.speed*self.rotation_vector_y

        if(self.border):
            self.coordinates[0][0] %= 100
            self.coordinates[0][1] %= 100



    def get2DpointList(self):
        return self.coordinates, [self.rotation_vector_x, self.rotation_vector_y]


class Ball(Object2D):
    
    def __init__(self, xPos, yPos):
        self.coordinates = np.array([[xPos, yPos],])

    def get2DpointList(self):
        return self.coordinates, None

class Render:

    colors = ["green", "red", "blue", "black"]
    cnumber = 4

    def __init__ (self, object2DList):
        self.objectList = object2DList
        self.window1 = graphics.GraphWin("window1", 400, 400)
        self.window1.setCoords(0, 0, 500, 500)
        self.graphicsObjectList = []
        self.text1 = graphics.Text(graphics.Point(1, 50), str(""))
        self.text2 = graphics.Text(graphics.Point(1, 50), str(""))

    def setObjects (self, object2DList):
        self.objectList = object2DList


    def renderFrame(self, reward = 0, episode = 0):
        for gObj in self.graphicsObjectList:
            gObj.undraw()
        self.graphicsObjectList = []

        textpos1 = graphics.Point(250, 50)
        self.text1.undraw()
        self.text1 = graphics.Text(textpos1, "Reward: " + str(reward))
        self.text1.draw(self.window1)
        textpos2 = graphics.Point(250, 450)
        self.text2.undraw()
        self.text2 = graphics.Text(textpos2, "Episode: " + str(episode))
        self.text2.draw(self.window1)
        i = 0
        for obj in self.objectList:
            points, rotation = obj.get2DpointList()
            if(rotation is not None):
                for point in points:
                    p1 = graphics.Point(point[0]+200-3*rotation[0], point[1]+200-3*rotation[1])
                    p2 = graphics.Point(point[0]+200+3*rotation[0], point[1]+200+3*rotation[1])
                    c1 = graphics.Circle(p1, 5)
                    c1.setFill(self.colors[i%self.cnumber])
                    c1.draw(self.window1)
                    c2 = graphics.Circle(p2, 5)
                    c2.setFill(self.colors[i % self.cnumber])
                    c2.draw(self.window1)
                    self.graphicsObjectList.append(c1)
                    self.graphicsObjectList.append(c2)

                i+=1
            else:
                for point in points:
                    p1 = graphics.Point(point[0]+200, point[1]+200)
                    c = graphics.Circle(p1, 7)
                    c.setFill(self.colors[i % self.cnumber])
                    c.draw(self.window1)
                    self.graphicsObjectList.append(c)
                i += 1

        #self.window1.getMouse() # Pause to view result
        # window.close()    # Close window when done


        



            


    
