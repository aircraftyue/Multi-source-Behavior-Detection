import logging
import time
import os
import cv2
# Import tf-pose-estimator from github (not tf-pose from python site-packages).
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from utils.actions import actionPredictor

# Log Setting.
logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Initialization.
CAMERA_INDEX = 0
MODEL_NAME = 'cmu'
SAVE_TXT_PATH = 'pose/joints.txt'  # Ubuntu
RESIZE_SCALE = '432x368'      # resize images before they are processed
RESIZE_OUT_RATIO = 4.0  # resize heatmaps before they are post-processed
fps_time = 0            # used to calculate fps.
i = 0                   # used to save img from camera.

# Save data.
def text_save(filename, data):
    """
    This is used to save data into file.
    """
    with open(filename, 'a') as file:   # Ubuntu
        file.write(str(data) + '\n')
    logger.info('saved as file.')

def save_joints(humans):
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
        num_part = human.part_count()
        
        # human.body_parts is a dict. key is the bodypart index, value is a BodyPart class data.
        for key, value in human.body_parts.items():
            idx = key
            x = round(value.x, 2)
            y = round(value.y, 2)
            # logger.info('Bodypart[{}]: ({:.2f}, {:.2f})'.format(idx, x, y))
            joints_dict[idx] = (x, y)
        # logger.debug('joints_dict:{}'.format(joints_dict))
        # e.g. joints_dict:{0: (0.44, 0.65), 14: (0.34, 0.52), 15: (0.53, 0.5), 16: (0.27, 0.55), 17: (0.65, 0.47)}

        # add one human's data into joints_humans list
        # BUG: multiple humans got same joints_dict. (sloved by adding joints_dict = {})
        joints_humans.append(joints_dict)
        # logger.debug('joints_humans:{}'.format(joints_humans))
        # clear joints_dict after appending to joints_humans list for saving next human data.
        # don't use joints_dict.clear(), cause it'll let joints_dict object be cleared.
        joints_dict = {}

    # (Optional) Save dicts into file.
    # text_save(SAVE_TXT_PATH, joints_humans)

    return joints_humans


# Lode Model and TfPose Estimator
logger.debug('initialization %s : %s' % (MODEL_NAME, get_graph_path(MODEL_NAME)))
w, h = model_wh(RESIZE_SCALE)
if w > 0 and h > 0:
    e = TfPoseEstimator(get_graph_path(MODEL_NAME), target_size=(w, h), trt_bool=False)
else:
    e = TfPoseEstimator(get_graph_path(MODEL_NAME), target_size=(432, 368), trt_bool=False)

# Analyze the coordinate of body part. Get actionPredictor object.
ap = actionPredictor()

# Open Camera
logger.info('cam read+')
cam = cv2.VideoCapture(CAMERA_INDEX)
ret_val, image = cam.read()
logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

while True:
    ret_val, image = cam.read()

    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=RESIZE_OUT_RATIO)
    # humans is a list with a single element, a string. 

    # Extract joints(joints) from humans.
    joints_humans = save_joints(humans) 
    # Analyze the coordinate of body part.
    statuses = ap.analyze_joints(image, joints_humans)
    # Draw skeleton.
    image, body_boxes = TfPoseEstimator.draw_skeleton(image, humans, statuses, imgcopy=False)


    cv2.putText(image,
                "FPS: %f" % (1.0 / (time.time() - fps_time)),
                (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)
    cv2.imshow('tf-pose-estimation result', image)
    fps_time = time.time()
    if (cv2.waitKey(1) & 0xFF) == 27: # ESC
        break
    if (cv2.waitKey(1) & 0xFF) == ord('s'):
        saved_status = cv2.imwrite('./images/saved/saved_%d.jpg'%i, image)
        logger.debug(f'cv2.imwrite: saved_status = {saved_status}')
        i += 1

cam.release()
cv2.destroyAllWindows()