"""
FYP

COMP 1682 - Final Year Project
Author: Omith Chowdhury
ID: 001236697

This one is for the sensors of the robot.

"""

import RPi.GPIO as GPIO
import time

# GPIO pin mapping for each ultrasonic sensor
SENSORS = {
    'front': {'trig': 17, 'echo': 27},
    'left':  {'trig': 22, 'echo': 23},
    'right': {'trig': 24, 'echo': 25},
}

# Setup GPIO mode
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

for sensor in SENSORS.values():
    GPIO.setup(sensor['trig'], GPIO.OUT)
    GPIO.setup(sensor['echo'], GPIO.IN)
    GPIO.output(sensor['trig'], False)

def read_distance(trig, echo):
    """
    Triggers a single HC-SR04 pulse and reads the echo to calculate distance.
    Returns: distance in metres (normalised 0.0–1.0, max 2.0 m).
    """
    # Trigger 10µs pulse
    GPIO.output(trig, True)
    time.sleep(0.00001)
    GPIO.output(trig, False)

    # Wait for echo to go high (start)
    start_time = time.time()
    while GPIO.input(echo) == 0:
        start_time = time.time()

    # Wait for echo to go low (end)
    end_time = time.time()
    while GPIO.input(echo) == 1:
        end_time = time.time()

    # Calculate duration and convert to distance
    elapsed = end_time - start_time
    distance_cm = (elapsed * 34300) / 2
    distance_m = distance_cm / 100.0

    # Normalise (clip to max 2.0 m → normalised 1.0)
    return min(distance_m / 2.0, 1.0)

def read_sonars():
    """
    Reads all three sensors (front, left, right) and returns normalised values.
    Output order must match PPO training: [front, left, right]
    """
    front = read_distance(**SENSORS['front'])
    left  = read_distance(**SENSORS['left'])
    right = read_distance(**SENSORS['right'])
    return front, left, right