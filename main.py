import os
import location
from pose import pose_monitor
from camera.ZedCamera import ZedCamera

if __name__ == '__main__':

    cam = ZedCamera()

    monitor = pose_monitor.PoseMonitor()
    monitor.run(cam)
