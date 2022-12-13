# Multi-source-Behavior-Detection

## 说明
功能：利用深度相机实现人体跌倒检测，并支持人体空间定位。  
检测判别有两个来源：人体关键点姿态分析，人体中心点三维坐标。  

若不使用深度相机做三维定位，仅用普通相机做跌倒检测，***可切换到 rgb-only 分支***，或退回代码版本至： ec62056be983df8945781df2a7258d0e69879db8  
其余细节可参考具体代码注释或git log。  

检测效果可参考B站视频（使用普通相机版本）：[【人体行为分析 + 跌倒检测  测试效果】 ](https://www.bilibili.com/video/BV1gK4y137B1/?share_source=copy_web&vd_source=ef958b90ac032df5a5067fa666506321)

---

人体关键点检测部分基于tf-pose实现：  
https://github.com/gsethi2409/tf-pose-estimation    

模型算法：  
https://github.com/CMU-Perceptual-Computing-Lab/openpose  

---

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

## 运行

连接好Zed相机后，首先进行相机标定，修改camera/ZedCamera.py中的参数。  

```python
python main.py
```

若检测视频文件，可修改pose_monitor.py中的 VIDEO_PATH 相关部分。
