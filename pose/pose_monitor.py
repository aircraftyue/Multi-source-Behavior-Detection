from datetime import datetime
import logging
import time
import cv2
# Import tf-pose-estimator from github (not tf-pose from python site-packages).
from pose.tf_pose.estimator import TfPoseEstimator
from pose.tf_pose.networks import get_graph_path, model_wh
from pose.utils.actions import actionPredictor

# Log Setting.
logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class PoseMonitor:
    """
    """
    def __init__(self):
        MODEL_NAME = 'cmu'
        self.SAVE_TXT_PATH = 'pose/joints.txt'  # Ubuntu
        RESIZE_SCALE = '432x368'      # resize images before they are processed
        self.RESIZE_OUT_RATIO = 4.0  # resize heatmaps before they are post-processed
        self.fps_time = 0            # used to calculate fps.
        self.i = 0                   # used to save img from camera.
        self.close_monitor = False
        VIDEO_PATH = './pose/videos/fall_50_ways.mp4' # for test

        # Load Model and TfPose Estimator
        logger.debug('initialization %s : %s' % (MODEL_NAME, get_graph_path(MODEL_NAME)))
        self.w, self.h = model_wh(RESIZE_SCALE)
        if self.w > 0 and self.h > 0:
            self.e = TfPoseEstimator(get_graph_path(MODEL_NAME), target_size=(self.w, self.h), trt_bool=False)
        else:
            self.e = TfPoseEstimator(get_graph_path(MODEL_NAME), target_size=(432, 368), trt_bool=False)

        # Analyze the coordinate of body part. Get actionPredictor object.
        self.ap = actionPredictor()

        # Open Camera
        # logger.info('cam read+')
        # if CAMERA_INDEX == "Zed":
        # #     self.cam = zed
        #     logger.info('Opening Zed Camera...')
        # elif CAMERA_INDEX is not None:
        #     self.cap = cv2.VideoCapture(CAMERA_INDEX)
        #     logger.info('Opening camera...')
        # else:
        #     self.cap = cv2.VideoCapture(VIDEO_PATH) 
        #     logger.info('Reading video...')
        # if not self.cap.isOpened():
        #     logger.error('Error opening video stream or file! ')


    def _text_save(self, filename, data):
        """
        This is used to save data into file.
        """
        with open(filename, 'a') as file:   # Ubuntu
            file.write(str(data) + '\n')
        logger.info('saved as file.')

    def _extract_joints(self, humans):
        """This is used to extract joints(body parts) data from skeleton.

        Return: joints_humans
        joints_humans: a list consist of each human's joints_dict
        joints_dict: a dict consist of each joint; key is idx, value is BodyPart(class, has part_idx, x, y...)
        idx: body part index(eg. 0 for nose)
        x, y: coordinate of body part (relative to the image size)
        """
        idx = 0
        x = 0
        y = 0
        # joints_dict : key is idx, value is (x, y)
        joints_dict = {}
        joints_humans = []
        for human in humans:
            # human.body_parts is a dict. key is the bodypart index, value is a BodyPart class data.
            for key, value in human.body_parts.items():
                idx = key
                x = round(value.x, 2)
                y = round(value.y, 2)
                joints_dict[idx] = (x, y)
            # e.g. joints_dict:{0: (0.44, 0.65), 14: (0.34, 0.52), 15: (0.53, 0.5), 16: (0.27, 0.55), 17: (0.65, 0.47)}

            # add one human's data into joints_humans list
            # BUG: multiple humans got same joints_dict. (sloved by adding joints_dict = {})
            joints_humans.append(joints_dict)
            # clear joints_dict after appending to joints_humans list for saving next human data.
            # don't use joints_dict.clear(), cause it'll let joints_dict object be cleared.
            joints_dict = {}

        # (Optional) Save dicts into file.
        # self._text_save(SAVE_TXT_PATH, joints_humans)

        return joints_humans

    def run(self, camera):

        frame_cnt = 0 # 控制帧率
        t_alert = time.time() # 记录alert时间并防止频繁报警

        while True:
            
            image, depth_image = camera.get_frame() # 获取ZedCamera帧数据

            frame_cnt = frame_cnt + 1
            if frame_cnt % 3 == 0: # 降低帧率为输入的1/3
                frame_cnt = 0 # 防止溢出

                ############################
                #===========核心===========#          
                # 使用模型预测人体
                humans = self.e.inference(image, resize_to_default=(self.w > 0 and self.h > 0), upsample_size=self.RESIZE_OUT_RATIO)
                # humans is a list with a single element, a string. 
                # 提起人体关键点
                joints_humans = self._extract_joints(humans) 
                # 分析姿态
                statuses, alert = self.ap.analyze_joints(image, joints_humans)
                # logger.debug(f'Statuses: {statuses}')
                #==========================# 
                ############################
                
                # alert屏蔽时间5s内只报一次
                if alert and (time.time() - t_alert > 3):
                    t_alert = time.time()   # 更新alert时间
                    t_str = datetime.now().strftime('%Y-%m-%d, %H:%M:%S')
                    logger.info(f"\n=====\nFALLING ALERT !!! \ntime: {t_str}\n=====\n")
                    # alert时保存一帧照片
                    cv2.imwrite("./output/fall_"+datetime.now().strftime('%Y%m%d_%H%M%S')+".jpg", image)
                    
                # Draw skeleton.
                image, body_boxes = TfPoseEstimator.draw_skeleton(image, humans, statuses, imgcopy=False)

                cv2.putText(image,
                            "FPS: %f" % (1.0 / (time.time() - self.fps_time)),
                            (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

                # TODO: GUI读取这里更新好的image,显示在GUI内部窗口区域上
                cv2.imshow('tf-pose-estimation result', image)
                self.fps_time = time.time()

            # 每帧检测控制按键
            if (cv2.waitKey(1) & 0xFF) == 27: # ESC
                break
            if (cv2.waitKey(1) & 0xFF) == ord('s'):
                saved_status = cv2.imwrite('./images/saved/saved_%d.jpg'%self.i, image)
                logger.debug(f'cv2.imwrite: saved_status = {saved_status}')
                self.i += 1

            if self.close_monitor:
                logger.info('Monitor is closing...')
                break

        self.cam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    print('Run Pose Monitor separately.')
    monitor = PoseMonitor(camera_index=0) # camera_index: None is video
    monitor.run()