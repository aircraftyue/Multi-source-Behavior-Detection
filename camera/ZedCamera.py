# -*- coding: utf-8 -*-
# @Time        :2022/6/24
# @Author      :yue
# @File        :ZedCamera.py
# @Description :封装zed摄像头

import os
import sys
import cv2
import pyzed.sl as sl
import numpy as np

class ZedCamera:
    def __init__(self):
        
        #####################
        ##    相机初始化
        #####################
        # Create Zed object
        self.zed = sl.Camera()

        # Config
        input_type = sl.InputType()
        if len(sys.argv) >= 2 :
            input_type.set_from_svo_file(sys.argv[1])
        init = sl.InitParameters(input_t=input_type)
        init.camera_resolution = sl.RESOLUTION.HD1080
        # init.depth_mode = sl.DEPTH_MODE.ULTRA
        init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        init.coordinate_units = sl.UNIT.MILLIMETER  

        # Open the camera
        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS :
            print(repr(err))
            self.zed.close()
            exit(1)

        # Set runtime parameters after opening the camera
        self.runtime = sl.RuntimeParameters()
        self.runtime.sensing_mode = sl.SENSING_MODE.STANDARD
        
        #####################
        ##    图像初始化
        #####################
        # Prepare new image size to retrieve half-resolution images
        self.scale = 0.5 # 用于图像放缩
        self.image_size = self.zed.get_camera_information().camera_resolution
        self.image_size.width = self.image_size.width * self.scale  # 显示画面的尺寸
        self.image_size.height = self.image_size.height * self.scale
        print(f"Zed | image size: {self.image_size.width}x{self.image_size.height}")
        
        # Declare your sl.Mat matrices
        self.image_zed = sl.Mat(self.image_size.width, self.image_size.height, sl.MAT_TYPE.U8_C4)
        self.depth_image_zed = sl.Mat(self.image_size.width, self.image_size.height, sl.MAT_TYPE.U8_C4)
        self.depth_map = sl.Mat(self.image_size.width, self.image_size.height)
        self.point_cloud = sl.Mat()
        
        #####################
        ##    内参矩阵
        #####################
        self.intrinsic_matrix_1080p = np.array([[1071.78831005611, 0, 973.549620870293],
                                                [0, 1068.37187888367, 565.348388162774],
                                                [0, 0, 1]])
        # TODO:标定720p
        self.intrinsic_matrix_720p = np.array([[None, 0, None],
                                                [0, None, None],
                                                [0, 0, 1]])
        
        #####################
        ##    外参矩阵
        #####################
        # 绕X轴旋转90°（逆时针为正）
        self.R_x90 = np.array([[1, 0, 0],
                               [0, 0, 1],
                               [0, -1, 0]])
        # 绕X轴旋转180°
        self.R_x180 = np.array([[1, 0, 0],
                                [0, -1, 0],
                                [0, 0, -1]])
        
        #================标定参数================#
        # Zed旋转矩阵： 使用 Zed Demo - position tracking 测量出的R(以拍摄位为终点)，取反后使用matlab函数eul2rotm转换得到
        self.external_RZed = np.array([[0.599017449319739, 0.403227077528133, -0.691799117778127],
                                    [-0.0905326328236623, 0.892519192553525, 0.441829529703052],
                                    [0.795601620036366, -0.202033202399321, 0.571139779146738]])
        # 相机标定位到世界坐标原点间平移：使用尺子测量出，单位m
        self.external_t = np.array([[0],[0],[2]])
        #========================================#
        
        # 旋转矩阵：从相机使用位，到相机标定位，再到世界坐标系的旋转
        self.external_R = self.R_x90.dot((self.R_x180.dot(self.external_RZed)).dot(self.R_x180))
        
        
    def calculate_coord_camera(self, coord_pixel, Zc):
        """将像素坐标转换到相机坐标系下

        Args:
            coord_pixel (元组): 像素坐标点(u,v)，单位为像素
            Zc (标量): 从Zed相机获取到的深度值，相机坐标系的z值

        Returns:
            标量: 相机坐标系的坐标点，单位为mm
        """
        u, v = coord_pixel
        coord_camera = Zc * np.matmul(np.linalg.inv(self.intrinsic_matrix_1080p), np.array([[u],[v],[1]]))
        
        Xc = coord_camera[0][0]
        Yc = coord_camera[1][0]
        
        return Xc, Yc, Zc
        
    # TODO：增加世界坐标系的转换
    def calculate_coord_world(self, coord_camera):
        """将像素坐标转换到相机坐标系下

        Args:
            coord_pixel (元组): 像素坐标点(u,v)，单位为像素
            Zc (标量): 从Zed相机获取到的深度值，相机坐标系的z值

        Returns:
            标量: 世界坐标系的坐标点，单位为m
        """
        Xc, Yc, Zc = coord_camera
        Xc /= 1000 # mm -> m
        Yc /= 1000
        Zc /= 1000 
        coord_world = np.matmul(self.external_R, np.array([[Xc],[Yc],[Zc]])) + self.external_t
        Xw = coord_world[0][0] 
        Yw = coord_world[1][0]
        Zw = coord_world[2][0]
        
        return Xw, Yw, Zw
        
    def get_frame(self):
        # 抓一帧左图和深度图
        err = self.zed.grab(self.runtime)
        if err == sl.ERROR_CODE.SUCCESS :
            # Retrieve the left image in the half-resolution
            self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT, sl.MEM.CPU, self.image_size)
            self.zed.retrieve_image(self.depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, self.image_size)
            # 深度测量图
            self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH, sl.MEM.CPU, self.image_size)
            # Retrieve the RGBA point cloud in half resolution
            # self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.GPU, self.image_size)

            # To recover data from sl.Mat to use it with opencv, use the get_data() method
            # It returns a numpy array that can be used as a matrix with opencv
            image_ocv = self.image_zed.get_data()
            depth_image_ocv = self.depth_image_zed.get_data() # shape: (360, 640, 4) 前三个通道是相同的深度图
            
            image = cv2.cvtColor(image_ocv, 1) # shape:(360,640,3)
            
        return image, depth_image_ocv, self.depth_map
        
    def get_distance(self, x, y):
        """获取指定图像坐标的三维深度
        
        Returns:
            距离相机的距离，单位为cm
        """
        # 抓一帧深度map
        # err = self.zed.grab(self.runtime)
        # if err == sl.ERROR_CODE.SUCCESS :
        #   self.zed.retrieve_image(self.depth_map, sl.MEASURE.DEPTH, sl.MEM.GPU, self.image_size)
        # 获取图像中坐标(x,y)处物体距离相机的距离
        err, depth_value = self.depth_map.get_value(x, y)
        if err == sl.ERROR_CODE.SUCCESS :
            print(f"Distance to Camera at ({x},{y}): {depth_value/10:.3f}cm")
            return depth_value/10
        
        print("Error | get_distance")
        return -1
    
    def get_camera_coord(self, x, y):
        """获取指定图像坐标的三维坐标
        图像坐标系 -> 相机坐标系
        
        Returns:
            Xc, Yc, Zc : 相机坐标系的坐标点，单位为mm
        """
        err, Zc = self.depth_map.get_value(x, y)
        coord_pixel = (x/self.scale, y/self.scale) # 用于匹配像素图的放缩
        return self.calculate_coord_camera(coord_pixel, Zc)
    
    def get_world_coord(self, x, y):
        """获取指定图像坐标的世界坐标系下的三维坐标
        图像坐标系 -> 相机坐标系 - > 世界坐标系
        
        Returns:
            Xc, Yc, Zc : 相机坐标系的坐标点，单位为m
        """
        coord_camera = self.get_camera_coord(x, y)
        return self.calculate_coord_world(coord_camera)
        
            

if __name__ == "__main__":

    cam = ZedCamera()
    h = int(cam.image_size.height)
    w = int(cam.image_size.width)
    
    while True:
        image, depth, _ = cam.get_frame()
        cv2.line(depth, pt1=(0,int(h/2)), pt2=(w,int(h/2)), color=(255,0,0), thickness=1) # X轴
        cv2.line(depth, pt1=(int(w/2),0), pt2=(int(w/2),h), color=(255,0,0), thickness=1) # Y轴
        for i in range(5):
            x = 200 + i*300
            for j in range(4):
                y = 100 + j*200
                Xc, Yc, Zc = cam.get_camera_coord(x,y)
                Xw, Yw, Zw = cam.get_world_coord(x,y)
                cv2.circle(depth, (x,y), radius=5, color=(255,255,0), thickness=-1)
                cv2.putText(depth, f"C({Xc:.1f},{Yc:.1f},{Zc:.1f})mm", (x,y), 
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0,0,255), thickness=1)
                cv2.putText(depth, f"W({Xw:.1f},{Yw:.1f},{Zw:.1f})m", (x,y+15), 
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255,0,255), thickness=1)
            # cv2.imshow("Image", image)
            cv2.imshow("Depth", depth)

        if (cv2.waitKey(1) & 0xFF) == 27: # ESC
            break
    

    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    cam.zed.close()
    
    print("\nFINISH")
        