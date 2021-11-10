#
#   Mevea client in Python
#   Connects REQ socket to tcp://localhost:5555
#

import zmq
import numpy as np
from math import sin

context = zmq.Context()

#  Socket to talk to server
print("Connecting to Mevea server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

# Initialize simulation time
simulationTime = 0
nInputs = 2
indexNo = np.array([float(x) for x in range(2)])

#  Do 1000 requests, waiting each time for a response
for request in range(1000):
    print("Sending request %s …" % request)

    # Set Input values
    #
    # Add your own control code here
    #
    
    indexNo[0] = sin(simulationTime)
    indexNo[1] = sin(simulationTime - 0.5)

    #
    #

    # Send control message
    inputValues = [f"{x:.17g}" for x in indexNo]
    controlMessage = " ".join(inputValues).encode()
    socket.send(controlMessage)

    #  Get the reply.
    sensorMessage:bytes = socket.recv()

    # assume message is a string of floats
    try:
        sensorValues = np.array([float(x) for x in sensorMessage.decode("utf-8").split(" ")])
    except:
        print("Error when converting floats in message")
        print(sensorMessage)
        socket.send(b"error")

    # Read the sensor values here 
    simulationTime = sensorValues[0]
    liftPosition = sensorValues[1]
    tiltPosition = sensorValues[2]
