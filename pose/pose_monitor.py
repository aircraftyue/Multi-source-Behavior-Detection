from datetime import datetime
import logging
import time
import cv2
# Import tf-pose-estimator from github (not tf-pose from python site-packages).
from pose.tf_pose.estimator import TfPoseEstimator
from pose.tf_pose.networks import get_graph_path, model_wh
from pose.utils.actions import actionPredictor

from camera.ZedCamera import ZedCamera
import pyzed.sl as sl

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
        self.pose_result = () # 汇总所有人姿态结果：(statues, locations)
        self.fall_result = () # 汇总跌倒检测结果：(alert, locations, time)

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

    def _text_save(self, filename, data):
        """
        用于保存数据
        """
        with open(filename, 'a') as file:   # Ubuntu
            file.write(str(data) + '\n')
        logger.info('saved as file.')

    def _extract_joints(self, humans):
        """从骨骼中提取关节数据(body parts)

        Return: joints_humans
        joints_humans: 包含每个人的 joints_dict 的列表
        joints_dict: 包含每个关节的字典; key is idx, value is BodyPart(class, has part_idx, x, y...)
        idx: body part index(eg. 0 for nose)
        x, y: body part的坐标 (相对于图像尺寸，是个比例)
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

    def get_location(self, image, human_centers, camera):
        """获得人体三维定位坐标

        Args:
            image : 二维图像
            human_centers : 人体中心点二维相对坐标
            camera : Zed深度相机

        Returns:
            locations: 相机坐标系下的三维坐标点
        """
        # 通过Zed相机获取深度
        locations = []
        for index, human_center in enumerate(human_centers):
            x, y = human_center  # x,y 为坐标比例(0.0, 0.0)~(1.0, 1.0)
            # 转换为像素坐标
            w, h = image.shape[1], image.shape[0]
            x, y = round(x*w), round(y*h)
            Xc, Yc, Zc  = camera.get_camera_coord(x, y)
            # print(f"Person[{index}] | Coord of Camera at ({x},{y}): ({Xc:.0f},{Yc:.0f},{Zc:.0f})")
            locations.append((Xc, Yc, Zc))
            # TODO: 测试 - 检测多人场景的人物匹配情况
        return locations

    def get_fall_result(self):
        """获取最新姿态情况

        Returns:
            元组: 列表-姿态字符串, 列表-所有人的三维坐标
        """
        return self.pose_result # statuses, locations
    
    def get_fall_result(self):
        """获取最新跌倒情况

        Returns:
            元组: 字符串-"Fall", 列表-所有跌倒人的三维坐标, 字符串-跌倒时间
        """
        return self.fall_result # "Fall", fall_locations, t_str

    def run(self, camera):

        frame_cnt = 0 # 控制帧率
        t_alert = time.time() # 记录alert时间并防止频繁报警
        
        h = int(camera.image_size.height)
        w = int(camera.image_size.width)

        while True:
            
            image, _, _ = camera.get_frame() # 获取ZedCamera帧数据

            frame_cnt = frame_cnt + 1
            if frame_cnt % 3 == 0: # 降低帧率为输入的1/3
                frame_cnt = 0 # 防止溢出

                #================核心================#          
                # 使用模型预测人体
                humans = self.e.inference(image, resize_to_default=(self.w > 0 and self.h > 0), upsample_size=self.RESIZE_OUT_RATIO)
                # 提起人体关键点
                joints_humans = self._extract_joints(humans) 
                # 分析姿态，返回人体姿态、跌倒警报、人体中心点
                statuses, alerts, human_centers = self.ap.analyze_joints(image, joints_humans)
                # 三维定位
                locations = self.get_location(image, human_centers, camera)
                #====================================# 


                #================图注================# 
                # 显示骨骼图
                image, body_boxes = TfPoseEstimator.draw_skeleton(image, humans, statuses, imgcopy=False)
                # 显示 FPS
                cv2.putText(image,
                            "FPS: %f" % (1.0 / (time.time() - self.fps_time)),
                            (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
                # 显示三维位置信息
                for index, (Xc, Yc, Zc) in enumerate(locations):
                    cv2.putText(image,
                            f"[{index}]Location: ({round(Xc/10,1)}, {round(Yc/10,1)}, {round(Zc/10,1)})cm",
                            (200, 10+30*index),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 0), 2)
                # 显示Xc, Yc 轴
                cv2.line(image, pt1=(0,int(h/2)), pt2=(w,int(h/2)), color=(255,0,0), thickness=1) # X轴
                cv2.line(image, pt1=(int(w/2),0), pt2=(int(w/2),h), color=(255,0,0), thickness=1) # Y轴
                #====================================#
                                
                                
                #================警报================# 
                # alert屏蔽时间5s内只报一次
                if (True in alerts) and (time.time() - t_alert > 5):
                    print(alerts)
                    fall_locations = []  # 保存所有跌倒人的坐标
                    # print(statuses) # statuses包含索引和状态信息['[0]Sit', '[1]Fall']
                    # 通过statuses判断谁跌倒了
                    for index, status in enumerate(statuses):
                        if "Fall" in status:
                            fall_locations.append(locations[index]) # 坐标单位为mm
                    t_alert = time.time()   # 更新alert时间
                    t_str = datetime.now().strftime('%Y-%m-%d, %H:%M:%S')
                    logger.info(f"\n=====\nFALLING ALERT !!! \ntime: {t_str}\n=====\n")
                    
                    # alert时保存一帧照片
                    cv2.imwrite("./output/fall_"+datetime.now().strftime('%Y%m%d_%H%M%S')+".jpg", image)
                    # 预留输出接口
                    self.fall_result = ("Fall", fall_locations, t_str)
                    print(self.fall_result)
                #====================================#  
                
                
                #================输出================#
                # TODO: GUI读取这里更新好的image,显示在GUI内部窗口区域上
                cv2.imshow('tf-pose-estimation result', image)
                self.pose_result = (statuses, locations)   # 所有人的坐标和状态
                #====================================# 
                
                self.fps_time = time.time()


            #================控制================#
            if (cv2.waitKey(1) & 0xFF) == 27: # ESC
                break
            if (cv2.waitKey(1) & 0xFF) == ord('s'):
                saved_status = cv2.imwrite('./images/saved/saved_%d.jpg'%self.i, image)
                logger.debug(f'cv2.imwrite: saved_status = {saved_status}')
                self.i += 1
            if self.close_monitor:
                logger.info('Monitor is closing...')
                break
            #====================================#
            
        self.cam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    print('Run Pose Monitor separately.')
    cam = ZedCamera()
    monitor = PoseMonitor()
    monitor.run(cam)