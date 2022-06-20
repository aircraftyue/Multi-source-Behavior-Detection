# python 3.x
# -*- coding:utf-8 -*-
# Author：Yue Zhen
# Date：2021-05-21
# Description：Analyze keypoints, then predicte human's action.

"""Analyze keypoints, then predicte human's action.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Main Method : 
  (1)人体髋关节中心点下降的速度大于预设的临界速度；
  (2)人体纵向中心线与地面的倾斜角小于预设的临界倾斜角；
  (3)人体外接矩形宽高比大于预设的临界比值。

  Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""
import math
import logging
import cv2
import numpy as np
from enum import IntEnum
import time

# Log Setting.
logger = logging.getLogger('actionPredictor')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class Joints(IntEnum):
    Nose = 0
    Neck = 1
    RElbow = 3
    LElbow = 6
    RHip = 8
    LHip = 11
    RKnee = 9
    LKnee = 12
    RAnkle = 10
    LAnkle = 13

# class Status(IntEnum):
#     STAND = 0
#     STAND_STILL = 1
#     STAND_WALKING = 2
#     FALL = 3
#     SIT = 4
#     LIE = 5

class actionPredictor:

    STAND = 0
    STAND_STILL = 1
    STAND_WALKING = 2
    FALL = 3
    SIT = 4
    LIE = 5

    # TODO:增加设置下列参数的接口
    HUMAN_ANGLE = 25
    MOVING_SPEED = 10
    ASPECT_RATIO = 0.6
    
    # used to calculate move speed.
    # TODO：区分不同人
    frame_time = 0           
    hip_position = (0, 0)   
    neck_position = (0, 0)     

    def __init__(self):
        # for multiple humans
        self.keypoints_humans = [{}]
        self.human_angles = []
        self.hip_positions = [(0,0)]
        self.neck_positions = [(0,0)]
        self.moving_speeds = []
        self.aspect_ratios = []
        self.speed_caches = [[0,0,0, 0,0,0, 0,0,0]]
        self.statuses = []
        self.MOVE_STATUS = {
            self.STAND:'Stand', 
            self.FALL:'Fall', 
            self.SIT:'Sit'
        }
        # used to alert.
        # 缓存越长，干扰越小
        self.statuses_cache = ['None','None','None','None','None','None']
        self.speed_cache = [0,0,0, 0,0,0, 0,0,0]  

    def __str__(self):
        return '[actionPredictor]'

    def get_key_joints(self,joints):
        return self.keypoints

    def get_status(self, joints):
        return self.statuses

    def joints_filter(self, joints):
        """过滤关节点，把重要的点提取出来

        joints is a dict
            eg. joints_dict:{0: (0.44, 0.65), 14: (0.34, 0.52), 15: (0.53, 0.5), 16: (0.27, 0.55), 17: (0.65, 0.47)}
        Return: keypoints(a dict), bool(用于判断是否可以使用)
        关键关节点[*最关键]：
        Nose = 0
        [*]Neck = 1
        [*]RHip = 8
        [*]LHip = 11
        RKnee = 9
        LKnee = 12
        [*]RAnkle = 10
        [*]LAnkle = 13
        """
        logger.debug(f'joints_filter: joints={joints}')
        keypoints = {}
        # keypoints 结构应如：{Joints.Nose:(0.52, 0.55), Joints.Neck:(0.56, 0.89), ...}
        keypoints[Joints.Nose] = joints.get(Joints.Nose, None)
        keypoints[Joints.Neck] = joints.get(Joints.Neck, None)
        keypoints[Joints.RElbow] = joints.get(Joints.RElbow, None)
        keypoints[Joints.LElbow] = joints.get(Joints.LElbow, None)
        keypoints[Joints.RHip] = joints.get(Joints.RHip, None)
        keypoints[Joints.LHip] = joints.get(Joints.LHip, None)
        keypoints[Joints.RKnee] = joints.get(Joints.RKnee, None)
        keypoints[Joints.LKnee] = joints.get(Joints.LKnee, None)
        keypoints[Joints.RAnkle] = joints.get(Joints.RAnkle, None)
        keypoints[Joints.LAnkle] = joints.get(Joints.LAnkle, None)
        
        logger.debug(f'joints_filter: keypoints={keypoints}')
        
        if(not (keypoints[Joints.Neck] and keypoints[Joints.RHip] and keypoints[Joints.LHip])) :
            logger.debug('joints_filter return False, cause we didn\'t find neck & hip.')
            return keypoints, False
            
        return keypoints, True
    

    def get_hip_position(self, keypoints):
        x_lhip, y_lhip = keypoints.get(Joints.LHip, None)
        x_rhip, y_rhip = keypoints.get(Joints.RHip, None)
        x_hip, y_hip = (x_lhip + x_rhip)/2, (y_lhip + y_rhip)/2
        return x_hip, y_hip 

    def get_neck_position(self, keypoints):
        x_neck, y_neck = keypoints.get(Joints.Neck, None)
        return x_neck, y_neck 
        
    # TODO：增加判断腿部夹角
    def human_orientation(self, keypoints, index):
        """计算人体纵向中心线的角度

        与12点钟方向相比较的夹角，此作为判断姿态的要素之一。
        先使用三点计算：Neck 和 两个Hip的
         - 躯干倾角可用于区分Fall和Sit
         - 腿部夹角可用于区分Stand和Sit
        """
        human_angle = 0
        x_neck, y_neck = self.get_neck_position(keypoints)
        x_hip, y_hip = self.get_hip_position(keypoints)
        dx = x_hip - x_neck
        dy = y_hip - y_neck

        # atan2的结果是与x轴正方向的夹角，范围是(-pi, pi)
        human_angle = round(((math.atan2(dy, dx) * 180) / math.pi)) - 90
        # human_angle的范围将是如下所示：
        #           0
        #           |
        #           |       
        # -90 —————————————— +90
        #           |       (-270)
        #           |
        #         -180
        logger.debug(f'human_orientation: human_angle = {human_angle}')
        return human_angle

    # TODO：求宽高比
    def body_box_calculation(self, keypoints, index):
        """计算人体外接框宽高比

        （walk around）由过滤后的关节点进行计算外接框，这不同于estimator中计算的外接框。
         - 可用于区分Stand和非Stand状态
        """
        aspect_ratio = 0

        points_coords = [point for key,point in keypoints.items() if (point != None)]
        # points_coords 应如 [(0.52, 0.55), (0.56, 0.89), ...]
        # 取x，y坐标最小和最大值
        x1 = min([point[0] for point in points_coords])
        y1 = min([point[1] for point in points_coords])
        x2 = max([point[0] for point in points_coords])
        y2 = max([point[1] for point in points_coords])

        # 计算宽高比, 
        box_w = round(abs(x2 - x1) * 100)
        box_h = round(abs(y2 - y1) * 100)
        aspect_ratio = box_w / box_h
        # 可以与 1.0 进行比较，当ratio明显小于1时说明是Stand状态，明显大于1时说明是非Stand状态。

        return aspect_ratio

    def speed_detection(self, keypoints, index):
        """计算人体移动速度

        return: moving_speed(float), moving_status(int, 1-移动, 0-静止)
         - 可用于识别状态变化，区分静卧和摔倒
         读取前一帧时间，然后获取当前时间，计算出时间间隔;
         读取前一帧髋关节坐标，获取当前坐标，计算出位移。
         TODO：区分移动和摔倒；摔倒的类型的思维导图
         如可以考虑比较neck和hip的速度差异，若neck速度远高于hip也对应摔倒
         或者仅计算主要关节的移动速度，作为状态变化因素，不直接影响决策

        """
        moving_speed = 0.0
        if(self.frame_time == 0):
            self.frame_time = time.time()

        # NOTE: 注意区分多人关节信息
        # 获取当前人体的关节
        x_hip, y_hip = self.get_hip_position(keypoints)
        x_neck, y_neck = self.get_neck_position(keypoints)
        # 取出上一帧的对应人体的关节，并保存当前关节
        if index < len(self.hip_positions):
            # 获取前一帧关节位置
            x_hip_old, y_hip_old = self.hip_positions[index]
            x_neck_old, y_neck_old = self.neck_positions[index]
            # 保存当前关节位置
            self.hip_positions[index] = (x_hip, y_hip)
            self.neck_positions[index] = (x_neck, y_neck)
        else:
            # 若index越界，表明此人是新检测到的
            # 以当前关节作为前一帧关节，将会导致delta_position为0
            x_hip_old, y_hip_old = x_hip, y_hip
            x_neck_old, y_neck_old = x_neck, y_neck
            # 并保存到多人关节列表中
            self.hip_positions.append((x_hip, y_hip))
            self.neck_positions.append((x_neck, y_neck))
            

        # TODO: 映射到真实环境的尺度上，方便调节速度参数
        # 考虑总体移动距离
        delta_position_hip = ((x_hip - x_hip_old) ** 2 + (y_hip - y_hip_old) ** 2) ** (1/2)
        delta_position_neck = ((x_neck - x_neck_old) ** 2 + (y_neck - y_neck_old) ** 2) ** (1/2)
        # 取hip和neck位移变化的均值，作为躯干的移动；两者经过上一步之后应该都是非负的
        delta_position = (max(delta_position_neck,0) + max(delta_position_hip,0)) / 2
        # 取两者差值，作为上半身角度变化速度（但可以由human_orientation部分实现），效果不好
        # delta_position = delta_position_neck - delta_position_hip

        # logger.debug(f'speed_detection: x_hip={x_hip}, x_hip_old={x_hip_old}, \
        #     y_hip={y_hip}, y_hip_old={y_hip_old}')
        # logger.debug(f'speed_detection: x_neck={x_neck}, x_neck_old={x_neck_old}, \
        #     y_neck={y_neck}, y_neck_old={y_neck_old}')   
        # logger.debug(f'speed_detection: delta_position_hip={delta_position_hip}, \
        #     delta_position_neck={delta_position_neck}, \
        #     delta_position={delta_position}')   

        # 计算两帧间时间差        
        frame_time_now = time.time()
        delta_time = frame_time_now - self.frame_time
        # 更新帧时间(不需要区分不同人)
        self.frame_time = frame_time_now

        # BUG：一个连续的人物运动过程中会多次进入
        # FIXED :这部分其实不需要判断delta_position为0，只要delta time不为0即可避免出错，而delta time正常情况下不会等于0；
        #        如果因为delta_position为0，直接返回0，0，则相当于废弃使用缓冲speed_caches，即直接导致moving_status发生改变；
        #        moving_status应该有缓冲区决定，而moving_speed可以实时变化。
        # if (delta_time == 0 or delta_position == 0):
        if (delta_time == 0):
            logger.debug(f'speed_detection: Wrong!!! delta_position={delta_position}, delta_time={delta_time}')
            return 0, 0

        # (delta_position_neck / time) - (delta_position_hip / time)
        moving_speed = delta_position / delta_time * 100
        self.frame_time = time.time()

        # logger.debug(f'speed_detection: moving_speed={moving_speed}, delta_position={delta_position}, delta_time={delta_time}')
        
        # BUG：moving_speed和moving_status会同时为0
        # FIXED：原因是上面因为判断delta_position为0，而直接return导致
        if index < len(self.speed_caches):
            if (moving_speed >= self.MOVING_SPEED):
                self.speed_caches[index].append('Moving')
            else:
                self.speed_caches[index].append('Still')
            # 为了保持cache长度不变，pop最旧的值
            self.speed_caches[index].pop(0)
            moving_times = self.speed_caches[index].count('Moving')
        else:
            # 增加一组cache
            self.speed_caches.append([0,0,0, 0,0,0, 0,0,0])
            moving_times = 0
        
        # if (moving_speed >= 1.0):
        #     self.speed_cache.append('Moving')
        # else:
        #     self.speed_cache.append('Still')
        # self.speed_cache.pop(0)
        # moving_times = self.speed_cache.count('Moving')

        if moving_times >= 2:
            moving_status = 1
        else:
            moving_status = 0

        logger.debug(f'speed_detection: speed_caches={self.speed_caches}, moving_status={moving_status}, moving_speed={moving_speed}')
        return moving_speed, moving_status
    
    def alert_decision(self, status):
        alert = False
        self.statuses_cache.append(status)
        self.statuses_cache.pop(0)
        fall_times = self.statuses_cache.count('Fall')
        if fall_times >= 3:
            alert = True
        logger.debug(f'alert_decision: fall_times={fall_times}, statuses_cache={self.statuses_cache}, alert={alert}')
        return alert

    # TODO:完善决策
    def action_analysis(self, human_angle, aspect_ratio, moving_speed=0, moving_status=0):
        """结合纵向中心线角度、髋关节下降速度、外接矩形框宽高比等信息，分析确定当前状态
        
        return：返回状态，字符串格式

        TODO 考虑加入膝关节与髋关节之间的夹角分析，以及下降速度
        区分：站立(静止/行走)，摔倒(过程)，静卧(状态)，坐(状态)
        可用参数：
            1. 人体倾角：可区分 stand,sit | fall, lie
            2. 外接框宽高比：stand | sit,fall | lie
            3. 移动速度：sit,lie,stand(still) | stand(walking),fall
            4. 移动状态（重要指标）：0-sit,lie,stand(still) | 1-stand(walking),fall
            5. TODO 状态切换：stand<>sit<>fall<>lie
        """

        # fall_score = 0.0
        # stand_score = 0.0
        # sit_score = 0.0
        # lie_score = 0.0
        # NOTE:排在前面的优先级高，相同分数时返回前者
        status_score = {
            'Stand_still':0.0, 
            'Stand_walking':0.0, 
            'Fall':0.0, 
            'Lie':0.0,
            'Sit':0.0, 
        }

        if human_angle in range(-self.HUMAN_ANGLE, self.HUMAN_ANGLE):
            status_score['Stand_still'] += 0.8
            status_score['Stand_walking'] += 0.8
            status_score['Sit'] += 0.8
        else:
            status_score['Fall'] += 0.8
            status_score['Lie'] += 0.8
        
        if aspect_ratio < self.ASPECT_RATIO:
            status_score['Stand_still'] += 0.8
            status_score['Stand_walking'] += 0.8
        elif aspect_ratio > 1/self.ASPECT_RATIO:
            status_score['Lie'] += 0.8
        else:
            status_score['Fall'] += 0.8
            status_score['Sit'] += 0.8

        if moving_speed < self.MOVING_SPEED:
            status_score['Stand_still'] += 0.8
            status_score['Stand_walking'] += 0.8
            status_score['Sit'] += 0.8
            status_score['Lie'] += 0.8
        else:
            status_score['Fall'] += 0.8
        
        if moving_status == 1:
            status_score['Fall'] += 1.0
            status_score['Stand_walking'] += 1.0
        else:
            status_score['Stand_still'] += 1.0
            status_score['Sit'] += 1.0
            status_score['Lie'] += 1.0


        score_max, status_max = max(zip(status_score.values(), status_score.keys()))

        # 不使用MOVE_STATUS了，直接返回key就好了

        # if fall_score > 1.5:
        #     # 'Fall'
        #     status = self.MOVE_STATUS[self.FALL]
        # else:
        #     # 'Stand'
        #     status = self.MOVE_STATUS[self.STAND]
        # # 'Sit
        # status = self.MOVE_STATUS[self.SIT]
        logger.debug(f'action_analysis: status_max = {status_max}, status_score={status_score}')
        return status_max

    # 放在TfPoseEstimator.draw_skeleton中去绘制了，这部分暂时不用了
    def draw_status(self, status, image, body_box):

        if body_box:
            # (body_x, body_y) is body center
            body_x = body_box['x']
            body_y = body_box['y']
            body_w = body_box['w']
            body_h = body_box['h']
            # coordinate used by cv2.rectangle has to be int.
            x = round(body_x - body_w / 2)
            y = round(body_y - body_h / 2)
            cv2.putText(np.array(image), status, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255), thickness=2)
        else:
            logger.debug('draw_status: body_box is None!')
        

    def analyze_joints(self, image, joints_humans):
        """
        joints_humans: a list, consist of joints(a dict) of each human
        body_box: x, y, w, h. (x, y) is center
        """
        # clear self.statuses list
        self.statuses = []
        # 此时传入的是多人数据，下面对每个人分别处理
        for index, joints in enumerate(joints_humans):
            # 首先预处理关节数据，生成自定的关节格式
            keypoints, keypoints_useful = self.joints_filter(joints)
            # 生成对应于多人joints_humans的预处理后的keypoints_humans，同样是多人数据，将保持索引不变
            self.keypoints_humans.append(keypoints)
            if (keypoints_useful):
                # 检测当前人体的信息，通过index对应不同人体
                # BUG：但若之前识别出的人此时未检测到，那么将导致index指向错误的人
                human_angle = self.human_orientation(keypoints, index)
                aspect_ratio = self.body_box_calculation(keypoints, index)
                moving_speed, moving_status = self.speed_detection(keypoints, index)
                cv2.putText(image,
                    "[{}]Speed: {:.2f}".format(index, moving_speed),
                    (10, 30+120*index),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 2)
                cv2.putText(image,
                    "[{}]Ratio: {:.2f}".format(index, aspect_ratio),
                    (10, 60+120*index),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 2)
                cv2.putText(image,
                    f"[{index}]Angle: {human_angle}",
                    (10, 90+120*index),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 2)
                cv2.putText(image,
                    f"[{index}]Moving Status: {moving_status}",
                    (10, 120+120*index),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 2)
                # logger.debug(f'analyze_joints: human_angle={human_angle}, aspect_ratio={aspect_ratio}, moving_speed={moving_speed}')
                # status = self.action_analysis(human_angle, moving_speed, aspect_ratio)
                status = self.action_analysis(human_angle, aspect_ratio, moving_speed, moving_status)
                alert = self.alert_decision(status)
                if(alert):
                    cv2.putText(image,
                        "FALLING ALERT!!!",
                        (180, 180),  cv2.FONT_HERSHEY_DUPLEX, 3,
                        (0, 0, 255), 5)
                self.statuses.append(f'{[index]}' + status)
            else:
                # 为了statuses内元素数量，与识别到的人数相匹配，避免出现索引越界情况
                status = None
                alert = self.alert_decision(status)
                if(alert):
                    cv2.putText(image,
                        "FALLING ALERT!!!",
                        (180, 180),  cv2.FONT_HERSHEY_DUPLEX, 3,
                        (0, 0, 255), 5)
                self.statuses.append(status)
                logger.debug('analyze_joints: keypoints is False!')
            
        return self.statuses

