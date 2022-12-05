# Multi-source-Behavior-Detection

## Pose Estimation
输入：RGB相机数据  
输出：行为估计结果，行走、站立、跌倒等

## 使用指导

核心代码在 pose/pose_monitor.py  
可参考注释进行修改  
支持空间定位的版本请切换至主分支main

## 关键依赖

```bash
tensorflow==2.5.0
tf-slim @ git+https://github.com/adrianc-a/tf-slim.git@80265a15482b4f81f162a344f13659a855cc5543

opencv-python
```
---
人体关键点检测部分基于tf-pose实现，迁移到了tf新版本：
https://github.com/gsethi2409/tf-pose-estimation

模型算法：
https://github.com/CMU-Perceptual-Computing-Lab/openpose

