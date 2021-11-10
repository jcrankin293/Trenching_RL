#
#   Mevea server in Python
#   Binds REP socket to tcp://*:5555
#
#   Jake Attempt

import zmq
import numpy as np

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

simulationTime = 0

def initScript():
    # Get the required GObjects from larger GDict set
    # This to make the script more efficient during run-time
    
    # Control signals by input name
    GObject.data['InputSlew'] = GDict['Input_Slew']
    GObject.data['InputBoomLift'] = GDict['Input_BoomLift']
    GObject.data['InputDipperArm'] = GDict['Input_DipperArm']
    GObject.data['InputBucket'] = GDict['Input_Bucket']

    # Displacement signals by DataSource name
    GObject.data['CylidnerBoomL'] = GDict['Cylinder_BoomLift_L']
    GObject.data['CylinderBoomR'] = GDict['Cylinder_BoomLift_R']
    GObject.data['CylinderDipper'] = GDict['Cylinder_DipperArm']
    GObject.data['CylinderBucket'] = GDict['Cylinder_Bucket']
    GObject.data['Slew'] = GDict['Slew']
    
    # Trench distances 
    GObject.data['DST1'] = GDict['DistanceSensorT1']
    GObject.data['DST2'] = GDict['DistanceSensorT2']
    GObject.data['DST3'] = GDict['DistanceSensorT3']
    GObject.data['DST4'] = GDict['DistanceSensorT4']
    GObject.data['DST5'] = GDict['DistanceSensorT5']
    GObject.data['DST6'] = GDict['DistanceSensorT6']
    GObject.data['DST7'] = GDict['DistanceSensorT7']
    GObject.data['DST8'] = GDict['DistanceSensorT8']

    # Reward signals
    GObject.data['DSDT'] = GDict['DistanceDiggerTrench']
    GObject.data['SoilTransfer2'] = GDict['SoilTransferSensor2']
    GObject.data['CollTrench'] = GDict['Collision_Trench']
    GObject.data['Fuel'] = GDict['AO_FuelConsumption']

    return 0

def callScript( deltaTime, simulationTime ):
  
    #  Wait for next request from client
    controlMessage:bytes = socket.recv()
    print(f"Message: {controlMessage}")

    # Assume message is a string of floats
    try:
        inputValues = np.array([float(x) for x in controlMessage.decode("utf-8").split(" ")])
    except:
        print("Error when converting floats in message")
        print(controlMessage)
        socket.send(b"error")
    
    # Set the values to input signals
    
    GObject.data['InputSlew'].setInputValue(inputValues[0])
    GObject.data['InputBoomLift'].setInputValue(inputValues[1])
    GObject.data['InputDipperArm'].setInputValue(inputValues[2])
    GObject.data['InputBucket'].setInputValue(inputValues[3])

    # Get sensor values from DataSorces
    # Displacement signals by DataSource name
    CylinderBoomL = GObject.data['CylidnerBoomL'].getDsValue()
    CylinderBoomR = GObject.data['CylinderBoomR'].getDsValue()
    CylidnerDipper = GObject.data['CylinderDipper'].getDsValue()
    CylinderBucket = GObject.data['CylinderBucket'].getDsValue()
    Slew = GObject.data['Slew'].getDsValue()
    
    # Get trench distances 
    DST1 = GObject.data['DST1'].getDsValue()
    DST2 = GObject.data['DST2'].getDsValue()
    DST3 = GObject.data['DST3'].getDsValue()
    DST4 = GObject.data['DST4'].getDsValue()
    DST5 = GObject.data['DST5'].getDsValue()
    DST6 = GObject.data['DST6'].getDsValue()
    DST7 = GObject.data['DST7'].getDsValue()
    DST8 = GObject.data['DST8'].getDsValue()

    # Get reward signals
    DSDT = GObject.data['DSDT'].getDsValue()
    SoilTransfer2 = GObject.data['SoilTransfer2'].getDsValue()
    CollTrench = GObject.data['CollTrench'].getDsValue()
    Fuel = GObject.data['Fuel'].getDsValue() 
    
    sensorValuesFromSolver = np.array([simulationTime, CylinderBoomL, CylinderBoomR, CylidnerDipper,CylinderBucket,DST1,DST2,DST3,DST4,DST5,DST6,DST7,DST8,DSDT,SoilTransfer2,CollTrench,Fuel])

    # Send sensor message
    sensorValues = [f"{x:.17g}" for x in sensorValuesFromSolver]
    messageBack = " ".join(sensorValues).encode()
    socket.send(messageBack)
    
    return 0
