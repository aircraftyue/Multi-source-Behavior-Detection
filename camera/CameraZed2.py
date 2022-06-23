#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        :2021/6/10 上午10:00
# @Author      :weiz
# @ProjectName :zed-samples
# @File        :CameraZed2.py
# @Description :封装一下zed摄像头
import cv2
import os
import time
import pyzed.sl as zed
import shutil
import math
 
 
class CameraZed2:
    def __init__(self, resolutionRatio=None, depthMode=None):
        self.cam = zed.Camera()
        self.input_type = zed.InputType()
        self.camInit = zed.InitParameters(input_t=self.input_type)
 
        # 设置分辨率
        if resolutionRatio == "HD2K":
            self.camInit.camera_resolution = zed.RESOLUTION.HD2K
        elif resolutionRatio == "HD1080":
            self.camInit.camera_resolution = zed.RESOLUTION.HD1080
        else:
            self.camInit.camera_resolution = zed.RESOLUTION.HD720
        # 设置获取深度信息的模式
        if depthMode == "PERFORMANCE":
            self.camInit.depth_mode = zed.DEPTH_MODE.PERFORMANCE
        elif depthMode == "QUALITY":
            self.camInit.depth_mode = zed.DEPTH_MODE.QUALITY
        else:
            self.camInit.depth_mode = zed.DEPTH_MODE.ULTRA
        self.camInit.coordinate_units = zed.UNIT.MILLIMETER  # 单位毫米
        err = self.cam.open(self.camInit)
        if err != zed.ERROR_CODE.SUCCESS:
            print(repr(err))
            self.cam.close()
            exit(1)
 
        self.image_size = self.cam.get_camera_information().camera_resolution
        self.image_zed = zed.Mat(self.image_size.width, self.image_size.height, zed.MAT_TYPE.U8_C4)
 
        # Set runtime parameters after opening the camera
        self.runtime = zed.RuntimeParameters()
        self.runtime.sensing_mode = zed.SENSING_MODE.STANDARD
 
    def dataStreamRunning(self):
        """
        用于获取摄像头的数据流
        :return:
        """
        image_zed_left = zed.Mat()  # left_img
        image_zed_right = zed.Mat()  # right_img
        depth_image = zed.Mat(self.image_size.width, self.image_size.height, zed.MAT_TYPE.U8_C4)
        self.point_cloud = zed.Mat()
        self.depth_map = zed.Mat()
 
        while True:
            self.cam.grab(self.runtime)
 
            # 左图
            self.cam.retrieve_image(image_zed_left, zed.VIEW.LEFT)
            image_cv_left = image_zed_left.get_data()
 
            # 右图
            self.cam.retrieve_image(image_zed_right, zed.VIEW.RIGHT)
            image_cv_right = image_zed_right.get_data()
 
            # 深度信息和点云
            self.cam.retrieve_image(depth_image, zed.VIEW.DEPTH, zed.MEM.CPU, self.image_size)
            self.cam.retrieve_measure(self.point_cloud, zed.MEASURE.XYZRGBA, zed.MEM.CPU, self.image_size)
            self.cam.retrieve_measure(self.depth_map, zed.MEASURE.DEPTH)
 
            self.image_cv_left = cv2.cvtColor(image_cv_left, 1)
            self.image_cv_right = cv2.cvtColor(image_cv_right, 1)
            self.image_depth = depth_image.get_data()
 
            yield
 
    def showImage(self):
        while True:
            next(self.dataStreamRunning())
 
            cv2.imshow("left", self.image_cv_left)
            cv2.imshow("right", self.image_cv_right)
            cv2.imshow("depth", self.image_depth)
            if cv2.waitKey(1) & 0xFF == 27:
                break
 
    def getMatImage(self, lr=None):
        """
        获取opencv格式的图片
        :param lr:获取左图或右图的标识位
        :return:
        """
        next(self.dataStreamRunning())
 
        if lr == "right":
            return self.image_cv_right
        else:
            return self.image_cv_left
 
    def getDepthValue(self, x, y):
        """
        获得某个像素点的深度信息
        :param x:
        :param y:
        :return:
        """
        next(self.dataStreamRunning())
        _, value = self.depth_map.get_value(x, y)
        return value
 
    def getPointCloud(self, x, y):
        """
        获得某个像素点的点云信息
        :param x:
        :param y:
        :return: [x, y, z, color],离左摄像头的距离
        """
        next(self.dataStreamRunning())
        _, point3D = self.point_cloud.get_value(x, y)
        distance = math.sqrt(point3D[0]*point3D[0] + point3D[1]*point3D[1] + point3D[2]*point3D[2])
        return point3D, distance
 
    def saveImageOfKey(self, savePath=None):
        """
        根据按键信息保存图片
        :param savePath:
        :return:
        """
        if savePath == None:
            savePath = "./zed2ImagesOfKey"
            if os.path.exists(savePath):
                shutil.rmtree(savePath)
            os.mkdir(savePath)
        else:
            if os.path.exists(savePath):
                shutil.rmtree(savePath)
            os.mkdir(savePath)
 
        num_l = 1
        num_r = 1
        while True:
            next(self.dataStreamRunning())
 
            cv2.imshow("left", self.image_cv_left)
            cv2.imshow("right", self.image_cv_right)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('l'):
                tmSavePath = os.path.join(savePath, "L{:0>3d}.png".format(num_l))
                cv2.imwrite(tmSavePath, self.image_cv_left)
                num_l = num_l + 1
            if key & 0xFF == ord('r'):
                tmSavePath = os.path.join(savePath, "R{:0>3d}.png".format(num_r))
                cv2.imwrite(tmSavePath, self.image_cv_right)
                num_r = num_r + 1
            if key & 0xFF == 27:
                break
 
    def saveImage(self, img, savePath=None, imageName=None):
        """
        保存图片，注意每秒只能保存一张图片
        :param savePath:
        :param imageName:
        :return:
        """
        if savePath == None:
            savePath = "./zed2Images"
        if imageName == None:
            imageName = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
 
        if not os.path.exists(savePath):
            os.mkdir(savePath)
        savePath = os.path.join(savePath, imageName + ".png")
 
        cv2.imwrite(savePath, img)
 

    def getOneFrame(self):
        """
        获取一帧的数据，默认使用左图
        :return: 左图cv格式图片
        """
        self.cam.grab(self.runtime)
        
        image_zed_left = zed.Mat()  # left_img
        self.cam.retrieve_image(image_zed_left, zed.VIEW.LEFT)
        image_cv_left = image_zed_left.get_data()
        self.image_cv_left = cv2.cvtColor(image_cv_left, 1)

        return self.image_cv_left
 
if __name__ == "__main__":
    cam = CameraZed2(depthMode="PERFORMANCE")
    # cam.showImage()
 
    # while True:
    #     img = cam.getMatImage(lr=)
    #     cv2.imshow("img", img)
    #     if cv2.waitKey(1) & 0xFF == 27:
    #         break
 
    # cam.saveImageOfKey()
    
    image = cam.getOneFrame()
    cv2.imshow("left", image)
    print(image.shape)
    cv2.waitKey(1)
 
    # img = cam.getMatImage()
    # cam.saveImage(img)
 
    print(cam.getDepthValue(100, 100))
 
    print(cam.getPointCloud(100, 100))