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
        ##    图片初始化
        #####################
        # Prepare new image size to retrieve half-resolution images
        self.image_size = self.zed.get_camera_information().camera_resolution
        self.image_size.width = self.image_size.width /2
        self.image_size.height = self.image_size.height /2
        print(f"Zed | image size: {self.image_size.width}x{self.image_size.width}")
        
        # Declare your sl.Mat matrices
        self.image_zed = sl.Mat(self.image_size.width, self.image_size.height, sl.MAT_TYPE.U8_C4)
        
    def get_frame(self):
    
        # 抓图像
        err = self.zed.grab(self.runtime)
        if err == sl.ERROR_CODE.SUCCESS :
            # Retrieve the left image in the half-resolution
            self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT, sl.MEM.CPU, self.image_size)

            # To recover data from sl.Mat to use it with opencv, use the get_data() method
            # It returns a numpy array that can be used as a matrix with opencv
            image_ocv = self.image_zed.get_data()
            image = cv2.cvtColor(image_ocv, 1)
            
        return image
        
        

if __name__ == "__main__":

    cam = ZedCamera()
    
    while True:
        image = cam.get_frame()
        cv2.imshow("Image", image)

        if (cv2.waitKey(1) & 0xFF) == 27: # ESC
            break
        