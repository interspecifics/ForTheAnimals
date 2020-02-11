"""
DuoEyes uses dual AMG8833 grideye breakout board
"""
 
import os
import math
import time
import busio
import board
import numpy as np
import pygame
from scipy.interpolate import griddata
from colour import Color 
import adafruit_amg88xx


i2c_bus = busio.I2C(board.SCL, board.SDA)

# lower (blue), higher (red)
MINTEMP = 14.
MAXTEMP = 25.
COLORDEPTH = 256

os.putenv('SDL_FBDEV', '/dev/fb1')
pygame.init()

# sensors
sensor_A = adafruit_amg88xx.AMG88XX(i2c_bus, 0x68)
sensor_B = adafruit_amg88xx.AMG88XX(i2c_bus, 0x69)
 
# pylint: disable=invalid-slice-index
points_A = [(math.floor(ix / 8), (ix % 8)) for ix in range(0, 64)]
grid_xA, grid_yA = np.mgrid[0:7:32j, 0:7:32j]
points_B = [(math.floor(ix / 8), (ix % 8)) for ix in range(0, 64)]
grid_xB, grid_yB = np.mgrid[0:7:32j, 0:7:32j]
# pylint: enable=invalid-slice-index
 
#sensor is an 8x8 grid so lets do a square
height = 480
width = 480
 
#the list of colors we can choose from
blue = Color("indigo")
colors = list(blue.range_to(Color("red"), COLORDEPTH))
colors = [(int(c.red * 255), int(c.green * 255), int(c.blue * 255)) for c in colors]

displayPixelWidth = width / 30
displayPixelHeight = height / 30

lcd = pygame.display.set_mode((2*width, height))
pygame.display.set_caption("{{.IR stereo vision.}}")

lcd.fill((255, 0, 0))

pygame.display.update()
pygame.mouse.set_visible(False)

lcd.fill((0, 0, 0))
pygame.display.update()

#some utility functions
def constrain(val, min_val, max_val):
    return min(max_val, max(min_val, val))
 
def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
 
#let the sensor initialize
time.sleep(.1)



 
while True: 
    #read the pixels
    pixels_A = []
    for row in sensor_A.pixels:
        pixels_A = pixels_A + row
    pixels_A = [map_value(p, MINTEMP, MAXTEMP, 0, COLORDEPTH - 1) for p in pixels_A]

    pixels_B = []
    for row in sensor_B.pixels:
        pixels_B = pixels_B + row
    pixels_B = [map_value(p, MINTEMP, MAXTEMP, 0, COLORDEPTH - 1) for p in pixels_B]

    #perform interpolation
    bicubic_A = griddata(points_A, pixels_A, (grid_xA, grid_yA), method='cubic')
    bicubic_B = griddata(points_B, pixels_B, (grid_xB, grid_yB), method='cubic')

    #draw everything
    for ix, row in enumerate(bicubic_A):
        for jx, pixel in enumerate(row):
            pygame.draw.rect(lcd, colors[constrain(int(pixel), 0, COLORDEPTH- 1)],
                             (displayPixelHeight * ix, displayPixelWidth * jx,
                              displayPixelHeight, displayPixelWidth))
    for ix, row in enumerate(bicubic_B):
        for jx, pixel in enumerate(row):
            pygame.draw.rect(lcd, colors[constrain(int(pixel), 0, COLORDEPTH- 1)],
                             (width + displayPixelHeight * ix, displayPixelWidth * jx,
                              displayPixelHeight, displayPixelWidth))
    pygame.display.update()
