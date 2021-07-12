#!/usr/bin/env python
# license removed for brevity

import os
import time
import numpy as np
import sys
sys.path.insert(0, '.')
import types
import rospy
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from cv_bridge.boost.cv_bridge_boost import getCvType

import torch
import torch.nn as nn

import lib.transform_cv2 as T
from lib.models import model_factory
from configs import set_cfg_from_file

torch.set_grad_enabled(False)
np.random.seed(123)


args = types.SimpleNamespace()
args.config             = '/home/cordin/catkin_ws/src/ROS_bisenet_segmentation/configs/bisenetv2_city.py'
args.weight_path        = '/home/cordin/catkin_ws/src/ROS_bisenet_segmentation/model_final_v2_city.pth'

cfg = set_cfg_from_file(args.config)

palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

# define model
net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='pred')
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
net.eval()
net.cuda()

# prepare data
to_tensor = T.ToTensor(
    mean=(0.3257, 0.3690, 0.3223), # city, rgb
    std=(0.2112, 0.2148, 0.2115),
)

def inference(frame):
    start_time = time.time()
    im = frame[:, :, ::-1]

    im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

    # inference
    out = net(im).squeeze().detach().cpu().numpy()
    pred = palette[out]
    elapsed_time = time.time() - start_time
    rospy.loginfo("Elapesed time: %s" % str(elapsed_time))
    
    return pred


def Image_to_opencv(msg):

    torch.cuda.empty_cache()
    cvb=CvBridge()
    cv_image = cvb.imgmsg_to_cv2(msg,"bgr8")

    inference_img = inference(cv_image)
    img_pub = rospy.Publisher("bisenet_inference_img", Image, queue_size=1)
    img_pub.publish(cvb.cv2_to_imgmsg(inference_img, "bgr8"))
    rospy.loginfo("complete pub")


if __name__ == '__main__':
    
    rospy.init_node("ROS_bisenet_inference", anonymous=True)
    rospy.loginfo("Running ROS_bisenet_inference")
    rospy.Subscriber("/usb_cam/image_raw", Image, Image_to_opencv, queue_size=1)

    rate=rospy.Rate(10)
    rate.sleep()
    
    rospy.spin()

