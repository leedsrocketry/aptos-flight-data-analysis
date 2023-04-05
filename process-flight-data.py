import matplotlib.pyplot as plt
import numpy as np
import sys
from ctypes import *
from dataclasses import dataclass, field
from typing import List

MessageCodes = [    "None",
                    "UnknownError",
                    "RecievedFailureFromMonitor",
                    "RecievedWatchdogFromMonitor",
                    "ArmFailed",
                    "ArmSuccess",
                    "LoggerError",
                    "GainUpdated",
                    "LaunchDetected",
                    "PrimaryWatchdogNotFed",
                    "MaxOrientationReached",
                    "StateChangedToStartup",
                    "StateChangedToOffLaunchRodTest",
                    "StateChangedToOffLaunchRodCalibration",
                    "StateChangedToIdle",
                    "StateChangedToLaunchRodCalibration",
                    "StateChangedToLaunchRodTest",
                    "StateChangedToArm",
                    "StateChangedToArmed",
                    "StateChangedToBelowVmc",
                    "StateChangedToActiveControlEnabled",
                    "StateChangedToDescent",
                    "StateChangedToShutdown",
                    "BTStartUpError",
                    "BTConnectionError",
                    "IMUStartUpError",
                    "BarometerStartUpError",
                    "SDTest"    ]

class PrimaryFlightDataPacket(Structure):
    _fields_ = [('elapsedTime', c_uint32),
                ('message', c_uint8),
                ('ax', c_float),
                ('ay', c_float),
                ('az', c_float),
                ('gx', c_float),
                ('gy', c_float),
                ('gz', c_float),
                ('altitude', c_float),
                ('verticalVelocity', c_float),
                ('w', c_float),
                ('x', c_float),
                ('y', c_float),
                ('z', c_float),
                ('wRate', c_float),
                ('xRate', c_float),
                ('yRate', c_float),
                ('zRate', c_float),
                ('roll', c_float),
                ('pitch', c_float),
                ('yaw', c_float),
                ('rollRate', c_float),
                ('pitchRate', c_float),
                ('yawRate', c_float),
                ('x1Target', c_float),
                ('x2Target', c_float),
                ('y1Target', c_float),
                ('y2Target', c_float),
                ('x1Actual', c_float),
                ('x2Actual', c_float),
                ('y1Actual', c_float),
                ('y2Actual', c_float),
                ('voltage', c_float)]

class MonitorFlightDataPacket(Structure):
    _fields_ = [('elapsedTime', c_uint32),
                ('message', c_uint8),
                ('ax', c_float),
                ('ay', c_float),
                ('az', c_float),
                ('gx', c_float),
                ('gy', c_float),
                ('gz', c_float),
                ('altitude', c_float),
                ('verticalVelocity', c_float),
                ('w', c_float),
                ('x', c_float),
                ('y', c_float),
                ('z', c_float),
                ('angleToVertical', c_float),
                ('servoReturnData', c_char * 30)]

@dataclass
class PrimaryFlightData():
    elapsedTime: list[float] = field(default_factory=list)
    message: list[float] = field(default_factory=list)
    ax: list[float] = field(default_factory=list)
    ay: list[float] = field(default_factory=list)
    az: list[float] = field(default_factory=list)
    gx: list[float] = field(default_factory=list)
    gy: list[float] = field(default_factory=list)
    gz: list[float] = field(default_factory=list)
    altitude: list[float] = field(default_factory=list)
    verticalVelocity: list[float] = field(default_factory=list)
    w: list[float] = field(default_factory=list)
    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    z: list[float] = field(default_factory=list)
    wRate: list[float] = field(default_factory=list)
    xRate: list[float] = field(default_factory=list)
    yRate: list[float] = field(default_factory=list)
    zRate: list[float] = field(default_factory=list)
    roll: list[float] = field(default_factory=list)
    pitch: list[float] = field(default_factory=list)
    yaw: list[float] = field(default_factory=list)
    rollRate: list[float] = field(default_factory=list)
    pitchRate: list[float] = field(default_factory=list)
    yawRate: list[float] = field(default_factory=list)
    x1Target: list[float] = field(default_factory=list)
    x2Target: list[float] = field(default_factory=list)
    y1Target: list[float] = field(default_factory=list)
    y2Target: list[float] = field(default_factory=list)
    x1Actual: list[float] = field(default_factory=list)
    x2Actual: list[float] = field(default_factory=list)
    y1Actual: list[float] = field(default_factory=list)
    y2Actual: list[float] = field(default_factory=list)
    voltage: list[float] = field(default_factory=list)

@dataclass
class MonitorFlightData():
    elapsedTime: list[float] = field(default_factory=list)
    message: list[float] = field(default_factory=list)
    ax: list[float] = field(default_factory=list)
    ay: list[float] = field(default_factory=list)
    az: list[float] = field(default_factory=list)
    gx: list[float] = field(default_factory=list)
    gy: list[float] = field(default_factory=list)
    gz: list[float] = field(default_factory=list)
    altitude: list[float] = field(default_factory=list)
    verticalVelocity: list[float] = field(default_factory=list)
    w: list[float] = field(default_factory=list)
    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    z: list[float] = field(default_factory=list)
    angleToVertical: list[float] = field(default_factory=list)
    servoReturnData: list[float] = field(default_factory=list)

# TODO: Set these to sensible values
MaximumAltitude_m = 1000
MaximumExpectedAcceleration_g = 15
MaximumExpectedRotationRate_rads = (2 * np.pi / 360) * 10
MaximumFinDeflectionAngle_deg = 15

def plotData(   figureTitle,
                flightLength_s,
                flightData,
                primaryData=True ):
    
    fig, axs = plt.subplots(2, 3, sharex=True)
    fig.subplots_adjust(hspace=0, wspace=0.4)
    fig.suptitle(figureTitle)

    # Position plots
    axs[0][0].plot(flightData.elapsedTime, flightData.altitude)
    axs[0][0].set_ylim(0, MaximumAltitude_m)
    axs[0][0].set_ylabel("Altitude (m)")

    axs[1][0].plot(flightData.elapsedTime, flightData.ax, label="x")
    axs[1][0].plot(flightData.elapsedTime, flightData.ay, label="y")
    axs[1][0].plot(flightData.elapsedTime, flightData.az, label="z")
    axs[1][0].set_ylim(-MaximumExpectedAcceleration_g, MaximumExpectedAcceleration_g)
    axs[1][0].set_ylabel("Body Frame Acceleration (G)")
    axs[1][0].legend()

    if primaryData:
        # Attitude plots
        axs[0][1].plot(flightData.elapsedTime, flightData.roll, label="roll")
        axs[0][1].plot(flightData.elapsedTime, flightData.pitch, label="pitch")
        axs[0][1].plot(flightData.elapsedTime, flightData.yaw, label="yaw")
        axs[0][1].set_ylim(-np.pi, np.pi)
        axs[0][1].set_ylabel("Attitude (rad)")
        axs[0][1].legend()

        axs[1][1].plot(flightData.elapsedTime, flightData.xRate, label="x")
        axs[1][1].plot(flightData.elapsedTime, flightData.yRate, label="y")
        axs[1][1].plot(flightData.elapsedTime, flightData.zRate, label="z")
        axs[1][1].plot(flightData.elapsedTime, flightData.wRate, label="w")
        axs[1][1].set_ylim(-MaximumExpectedRotationRate_rads, MaximumExpectedRotationRate_rads)
        axs[1][1].set_ylabel("Attitude Rates (rad/s)")
        axs[1][1].legend()

        # Fin deflection plots
        axs[0][2].plot(flightData.elapsedTime, flightData.x1Target, label="x1")
        axs[0][2].plot(flightData.elapsedTime, flightData.x2Target, label="x2")
        axs[0][2].plot(flightData.elapsedTime, flightData.y1Target, label="y1")
        axs[0][2].plot(flightData.elapsedTime, flightData.y2Target, label="y1")
        axs[0][2].set_ylim(-MaximumFinDeflectionAngle_deg, MaximumFinDeflectionAngle_deg)
        axs[0][2].set_ylabel("Target Fin Deflections (deg)")
        axs[0][2].legend()

        axs[1][2].plot(flightData.elapsedTime, (flightData.x1Target - flightData.x1Actual), label="x1")
        axs[1][2].plot(flightData.elapsedTime, (flightData.x2Target - flightData.x2Actual), label="x2")
        axs[1][2].plot(flightData.elapsedTime, (flightData.y1Target - flightData.y1Actual), label="y1")
        axs[1][2].plot(flightData.elapsedTime, (flightData.y2Target - flightData.y2Actual), label="y2")
        axs[1][2].set_ylim(-90, 90)
        axs[1][2].set_ylabel("Measured Fin Deflection Error (deg)")
        axs[1][2].legend()
    
    else:
        # Vertical velocity plot
        axs[0][1].plot(flightData.elapsedTime, flightData.verticalVelocity)
        #axs[0][1].set_ylim(-np.pi, np.pi)
        axs[0][1].set_ylabel("Vertical Velocity (m/s)")

        # Angle to vertical plot
        axs[1][1].plot(flightData.elapsedTime, flightData.angleToVertical)
        #axs[1][1].set_ylim(-np.pi, np.pi)
        axs[1][1].set_ylabel("Angle to Vertical (deg)")

    # Make changes to all plots
    for ax in axs.flatten():
        ax.grid()
        ax.axvline(x=0, color='g', linestyle='--')
        ax.set_xlim(0, flightLength_s)
    
    # Add x-axis label to all the bottom row subplots
    for ax in axs[-1]:
        ax.set_xlabel("Time (T + Xs)")

def parseRawFlightData(flightDateFilename, primaryData=True):
    if primaryData:
        packet = PrimaryFlightDataPacket()
        flightData = PrimaryFlightData()
    
    else:
        packet = MonitorFlightDataPacket()
        flightData = MonitorFlightData()
    
    # Populate the flightData with teh values from a series of decoded packets
    with open(flightDateFilename, "rb") as flightDataFile:
        while flightDataFile.readinto(packet) == sizeof(packet):
            for field, fieldType in packet._fields_:
                value = getattr(packet, field)
                getattr(flightData, field).append(value)
    
    # Replace message codes with their literal equivalents
    for i in range(0, len(flightData.message)):
        flightData.message[i] = MessageCodes[flightData.message[i]]
    
    # Convert all flightData variables to np arrays (makes operating on them easier in future)
    for variable in flightData.__dict__.keys():
        value = getattr(flightData, variable)
        setattr(flightData, variable, np.array(value))
    
    return flightData

def correctTimes(flightData, flightLength_s):
    # Convert elapsedTime microseconds to seconds
    flightData.elapsedTime = flightData.elapsedTime / 10**6

    # Convert elapsedTime to a cummulative value
    for i in range(1, len(flightData.elapsedTime)):
        flightData.elapsedTime[i] = flightData.elapsedTime[i - 1] + flightData.elapsedTime[i]
    
    launchDetected = False
    flightEndIndex = -1

    # Find the time of the first datapoint after launch and the final datapoint
    for i in range(0, len(flightData.elapsedTime)):
        elapsedTime = flightData.elapsedTime[i]
        message = flightData.message[i]

        # Need to ignore if the launch was detected multiple times
        if not launchDetected:
            if (message == "LaunchDetected"):
                launchDetected = True

                # Subtract the current time from all values to make this T-0
                flightData.elapsedTime -= elapsedTime
        
        # flightData.elapsedTime has to be adjusted for the new T-0 time before this can be considered
        else:
            if (elapsedTime > flightLength_s):
                flightEndIndex = i
                break
    
    # Trim the data to the length of the flight
    for variable in flightData.__dict__.keys():
        setattr(flightData, variable, getattr(flightData, variable)[0:flightEndIndex])
    
    return flightData
    
def main():
    # Example usage: python process-flight-data.py example-aptos-primary-log.dat example-aptos-monitor-log.dat 1 01/04/23 1000
    primaryFlightDateFilename = sys.argv[1]
    monitorFlightDateFilename = sys.argv[2]
    flightNumber = int(sys.argv[3])
    flightDate = sys.argv[4]
    flightLength_s = int(sys.argv[5])

    primaryFlightData = parseRawFlightData(primaryFlightDateFilename)
    monitorFlightData = parseRawFlightData(monitorFlightDateFilename, primaryData=False)

    primaryFlightData = correctTimes(primaryFlightData, flightLength_s)
    monitorFlightData = correctTimes(monitorFlightData, flightLength_s)

    print(f"Unique Primary Flight Messages: %s" % np.unique(primaryFlightData.message))
    print(f"\nUnique Monitor Flight Messages: %s" % np.unique(monitorFlightData.message))
    
    plotData(f"Aptos Flight %i (%s) Data (%s)" % (flightNumber, flightDate, "Primary"), flightLength_s, primaryFlightData)
    plotData(f"Aptos Flight %i (%s) Data (%s)" % (flightNumber, flightDate, "Monitor"), flightLength_s, monitorFlightData, primaryData=False)

    plt.show()

if __name__ == "__main__":
    main()