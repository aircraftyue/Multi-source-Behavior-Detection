# Multi-source-Behavior-Detection

## Main
1. Pose： 调用摄像头实时输出行为估计结果
2. Camera： Zed双目相机，可输出深度图

## Pose Estimation
输入：相机单目数据  
输出：姿态估计结果，跌倒警报


## 3D-Location
输入：相机深度图，人体关键点坐标  
输出：三维空间人体定位坐标

## Requirements
关键包：
```bash
tensorflow==2.5.0
tf-slim @ git+https://github.com/adrianc-a/tf-slim.git@80265a15482b4f81f162a344f13659a855cc5543

opencv-python
pyzed   # 需要通过Zed-SDK安装
```
