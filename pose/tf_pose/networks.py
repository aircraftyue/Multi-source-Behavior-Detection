import os
from os.path import dirname, abspath

import tensorflow as tf

# from tf_pose.network_mobilenet import MobilenetNetwork
# from pose.tf_pose.network_mobilenet_thin import MobilenetNetworkThin
from pose.tf_pose import network_base

# from tf_pose.network_cmu import CmuNetwork
# from tf_pose.network_mobilenet_v2 import Mobilenetv2Network


def _get_base_path():
    if not os.environ.get('OPENPOSE_MODEL', ''):
        return './models'
    return os.environ.get('OPENPOSE_MODEL')

# # 训练时使用
# def get_network(type, placeholder_input, sess_for_load=None, trainable=True):
#     if type == 'mobilenet_thin':
#         net = MobilenetNetworkThin({'image': placeholder_input}, conv_width=0.75, conv_width2=0.50, trainable=trainable)
#         pretrain_path = 'pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224.ckpt'
#         last_layer = 'MConv_Stage6_L{aux}_5'
#     elif type in ['cmu', 'openpose']:
#         net = CmuNetwork({'image': placeholder_input}, trainable=trainable)
#         pretrain_path = 'numpy/openpose_coco.npy'
#         last_layer = 'Mconv7_stage6_L{aux}'

#     else:
#         raise Exception('Invalid Model Name.')

#     pretrain_path_full = os.path.join(_get_base_path(), pretrain_path)
#     if sess_for_load is not None:
#         if type in ['cmu', 'vgg', 'openpose']:
#             if not os.path.isfile(pretrain_path_full):
#                 raise Exception('Model file doesn\'t exist, path=%s' % pretrain_path_full)
#             net.load(os.path.join(_get_base_path(), pretrain_path), sess_for_load)
#         else:
#             try:
#                 s = '%dx%d' % (placeholder_input.shape[2], placeholder_input.shape[1])
#             except:
#                 s = ''
#             ckpts = {
#                 'mobilenet': 'trained/mobilenet_%s/model-246038' % s,
#                 'mobilenet_thin': 'trained/mobilenet_thin_%s/model-449003' % s,
#                 'mobilenet_fast': 'trained/mobilenet_fast_%s/model-189000' % s,
#                 'mobilenet_accurate': 'trained/mobilenet_accurate/model-170000',
#                 'mobilenet_v2_w1.4_r0.5': 'trained/mobilenet_v2_w1.4_r0.5/model_latest-380401',
#                 'mobilenet_v2_large': 'trained/mobilenet_v2_w1.4_r1.0/model-570000',
#                 'mobilenet_v2_small': 'trained/mobilenet_v2_w0.5_r0.5/model_latest-380401',
#             }
#             ckpt_path = os.path.join(_get_base_path(), ckpts[type])
#             loader = tf.train.Saver()
#             try:
#                 loader.restore(sess_for_load, ckpt_path)
#             except Exception as e:
#                 raise Exception('Fail to load model files. \npath=%s\nerr=%s' % (ckpt_path, str(e)))

#     return net, pretrain_path_full, last_layer


def get_graph_path(model_name):
    dyn_graph_path = {
        'cmu': 'graph/cmu/graph_opt.pb',
        'mobilenet_thin': 'graph/mobilenet_thin/graph_opt.pb',
        'mobilenet_v2_large': 'graph/mobilenet_v2_large/graph_opt.pb',
    }

    base_data_dir = dirname(dirname(abspath(__file__)))
    if os.path.exists(os.path.join(base_data_dir, 'models')):
        base_data_dir = os.path.join(base_data_dir, 'models')
    else:
        base_data_dir = os.path.join(base_data_dir, 'tf_pose_data')

    graph_path = os.path.join(base_data_dir, dyn_graph_path[model_name])
    if os.path.isfile(graph_path):
        return graph_path

    raise Exception('Graph file doesn\'t exist, path=%s' % graph_path)


def model_wh(resolution_str):
    width, height = map(int, resolution_str.split('x'))
    if width % 16 != 0 or height % 16 != 0:
        raise Exception('Width and height should be multiples of 16. w=%d, h=%d' % (width, height))
    return int(width), int(height)
