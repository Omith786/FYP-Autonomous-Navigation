"""
FYP
COMP 1682 - Final Year Project
Author: Omith Chowdhury
ID: 001236697   

This one is for the motors of the robot.

Using library from Freenove, the company that made the robot kit.

PPO Actions:
- 0: move forward
- 1: turn left (rotate in place)
- 2: turn right (rotate in place)

For simplicity, movement duration and speed are fixed per step.

"""

from Freenove_Direct_Motor import MotorDriver
import time

# Initialise Freenove motor driver (adjust parameters if needed)
motor = MotorDriver()

# Global movement parameters
SPEED = 80           # Speed value (0–100)
DURATION = 0.15      # Duration of each action in seconds

def move_forward():
    """
    Moves the robot forward in a straight line for a short time step.
    """
    motor.move(SPEED, SPEED)
    time.sleep(DURATION)
    motor.stop()

def turn_left():
    """
    Rotates the robot left in place (clockwise on one wheel, opposite on the other).
    """
    motor.move(-SPEED, SPEED)
    time.sleep(DURATION)
    motor.stop()

def turn_right():
    """
    Rotates the robot right in place (counter-clockwise on one wheel, opposite on the other).
    """
    motor.move(SPEED, -SPEED)
    time.sleep(DURATION)
    motor.stop()
