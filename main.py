import location
from pose import pose_monitor

if __name__ == '__main__':

    print('Run Pose Monitor separately.')
    monitor = pose_monitor.PoseMonitor(camera_index=0)
    monitor.run()
