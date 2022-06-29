# -*- coding: utf-8 -*-
# @Time        :2022/6/24
# @Author      :yue
# @File        :ZedCamera.py
# @Description :封装zed摄像头

import os
import sys
import cv2
import pyzed.sl as sl

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
        init.camera_resolution = sl.RESOLUTION.HD720
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
        self.image_size = self.zed.get_camera_information().camera_resolution
        self.image_size.width = self.image_size.width /2
        self.image_size.height = self.image_size.height /2
        print(f"Zed | image size: {self.image_size.width}x{self.image_size.width}")
        
        # Declare your sl.Mat matrices
        self.image_zed = sl.Mat(self.image_size.width, self.image_size.height, sl.MAT_TYPE.U8_C4)
        self.depth_image_zed = sl.Mat(self.image_size.width, self.image_size.height, sl.MAT_TYPE.U8_C4)
        self.depth_map = sl.Mat(self.image_size.width, self.image_size.height)
        self.point_cloud = sl.Mat()
        
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
            

if __name__ == "__main__":

    cam = ZedCamera()
    
    while True:
        image, depth = cam.get_frame()
        for i in range(3):
            x = 100 + i*200
            for j in range(3):
                y = 50 + j*100
                depth_value = cam.get_distance(x,y)
                cv2.circle(depth, (x,y), radius=5, color=(255,255,0), thickness=-1)
                cv2.putText(depth, f"{depth_value:.1f}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255,0,255), thickness=1)
            cv2.imshow("Image", image)
            cv2.imshow("Depth", depth)

        if (cv2.waitKey(1) & 0xFF) == 27: # ESC
            break
    

    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    cam.zed.close()
    
    print("\nFINISH")
        