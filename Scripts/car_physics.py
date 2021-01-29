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
        self.coordinates = np.array([[x_pos, y_pos],])
        self.speed = 0

        print("Car initialized.")
        print("rotation_step_size = " + str(self.rotation_step_size))
        print("maxspeed = " + str(maxspeed))
        print("joint coordinates:" +str(self.coordinates))
    
    def move(self, gaspedal = 0, rotate = 0):

        # If gaspedal is pushed accelerate, else deccelerate
        if(gaspedal!=0):
            self.speed+=(self.acceleration*gaspedal)
        elif(self.speed>0):
            self.speed-=(self.acceleration)

        # Make sure speed is not negative or above max speed or 0: 
        if(self.speed > self.maxspeed):
            self.speed = self.maxspeed
        elif(self.speed < 0):
            self.speed = 0
            
        if(rotate==0):
            self.rotation -= self.rotation_step_size
        elif(rotate==2):
            self.rotation += self.rotation_step_size
        elif(rotate!=1):
            print("invalid rotate value! Must be -1, 0 or 1!")

        self.coordinates[0][0] += self.speed*np.sin(self.rotation)
        self.coordinates[0][1] += self.speed*np.cos(self.rotation)
        
        if(self.border):
            self.coordinates[0][0] %= 100
            self.coordinates[0][1] %= 100



    def get2DpointList(self):
        return self.coordinates


class Ball(Object2D):
    
    def __init__(self, xPos, yPos):
        self.coordinates = np.array([[xPos, yPos],])

    def get2DpointList(self):
        return self.coordinates

class Render:

    colors = ["green", "red", "blue", "black"]
    cnumber = 4

    def __init__ (self, object2DList):
        self.objectList = object2DList
        self.window1 = graphics.GraphWin("window1", 800, 800)
        self.window1.setCoords(0, 0, 500, 500)
        self.graphicsObjectList = []
        self.text = graphics.Text(graphics.Point(1, 50), str(""))

    def setObjects (self, object2DList):
        self.objectList = object2DList


    
    def renderFirstFrame(self, reward = 0):
        for gObj in self.graphicsObjectList:
            gObj.undraw()
        self.graphicsObjectList = []

        textpos = graphics.Point(500, 50)
        self.text.undraw()
        self.text = graphics.Text(textpos, str(reward))
        self.text.draw(self.window1)
        i = 0
        for obj in self.objectList:
            points = obj.get2DpointList()
            for point in points:
                p1 = graphics.Point(point[0]+200, point[1]+200)
                c = graphics.Circle(p1, 7)
                c.setFill(self.colors[i%self.cnumber])
                c.draw(self.window1)
                self.graphicsObjectList.append(c)
            i+=1

    def renderFrame(self, reward = 0):
        for gObj in self.graphicsObjectList:
            gObj.undraw()
        self.graphicsObjectList = []

        textpos = graphics.Point(200, 50)
        self.text.undraw()
        self.text = graphics.Text(textpos, str(reward))
        self.text.draw(self.window1)
        i = 0
        for obj in self.objectList:
            points = obj.get2DpointList()
            for point in points:
                p1 = graphics.Point(point[0]+200, point[1]+200)
                c = graphics.Circle(p1, 7)
                c.setFill(self.colors[i%self.cnumber])
                c.draw(self.window1)
                self.graphicsObjectList.append(c)
            i+=1

        #self.window1.getMouse() # Pause to view result
        # window.close()    # Close window when done


        



            


    
