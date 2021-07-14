#!/usr/bin/env python
# license removed for brevity

import os
import time
import numpy as np
import sys
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

sys.path.insert(0, '/home/cordin/catkin_ws/src/ROS_bisenet_segmentation')
torch.set_grad_enabled(False)
np.random.seed(123)

mapping = { 
        0: 19,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6, 
        7: 7, 
        8: 8,
        9: 9,
        10: 10,
        11: 0, ## human
        12: 0, ## rider
        13: 0, ## car
        14: 0, ## truck
        15: 0, ## bus
        16: 0, ## train
        17: 0, ## motorcycle
        18: 0, ## bicycle
        -1: 0,
        255: 0
    }

def encode_labels(mask):
    label_mask = np.zeros_like(mask)
    for k in mapping:
        label_mask[mask == k] = mapping[k]
    return label_mask

args = types.SimpleNamespace()
args.config             = '/home/cordin/catkin_ws/src/ROS_bisenet_segmentation/configs/bisenetv2_city.py'
args.weight_path        = '/home/cordin/catkin_ws/src/ROS_bisenet_segmentation/model_final_v2_city.pth'

cfg = set_cfg_from_file(args.config)

palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

orb = cv2.ORB_create(
    nfeatures=1000,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=31,
    firstLevel=0,
    WTA_K=2,
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=20,
)

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
    fr1 = frame.copy()
    fr2 = frame.copy()
    im = frame[:, :, ::-1]

    im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

    # inference
    out = net(im).squeeze().detach().cpu().numpy()

    pred = np.where(out==0,19,out)
    pred = palette[pred]
    out = encode_labels(out)
    pred_dynamic = palette[out]

    elapsed_time = time.time() - start_time
    rospy.loginfo("Elapesed time: %s" % str(elapsed_time))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    base_mask = np.array(out, dtype=np.uint8) 
    
    kp1, des1 = orb.detectAndCompute(gray, base_mask)
    kp2, des2 = orb.detectAndCompute(gray, None)

    for i in range(len(kp1)):
        idx = i
        x1, y1 = kp1[idx].pt
        cv2.circle(fr1, (int(x1), int(y1)), 3, (255, 0, 0), 3)
    
    for i in range(len(kp2)):
        idx = i
        x1, y1 = kp2[idx].pt
        cv2.circle(fr2, (int(x1), int(y1)), 3, (255, 0, 0), 3)

    return pred, pred_dynamic, fr1, fr2


def Image_to_opencv(msg):

    torch.cuda.empty_cache()
    cvb=CvBridge()
    cv_image = cvb.imgmsg_to_cv2(msg,"bgr8")

    inference_img,inference_img_dynamic, fr1, fr2 = inference(cv_image)

    img_pub = rospy.Publisher("bisenet_inference_img", Image, queue_size=1)
    img_pub_dynamic = rospy.Publisher("bisenet_inference_img_dynamic", Image, queue_size=1)
    img_pub_fr1 = rospy.Publisher("bisenet_inference_img_fr1", Image, queue_size=1)
    img_pub_fr2 = rospy.Publisher("bisenet_inference_img_fr2", Image, queue_size=1)

    img_pub.publish(cvb.cv2_to_imgmsg(inference_img, "bgr8"))
    img_pub_dynamic.publish(cvb.cv2_to_imgmsg(inference_img_dynamic, "bgr8"))
    img_pub_fr1.publish(cvb.cv2_to_imgmsg(fr1, "bgr8"))
    img_pub_fr2.publish(cvb.cv2_to_imgmsg(fr2, "bgr8"))
    rospy.loginfo("complete pub")


if __name__ == '__main__':
    
    rospy.init_node("ROS_bisenet_inference", anonymous=True)
    rospy.loginfo("Running ROS_bisenet_inference")
    rospy.Subscriber("/usb_cam/image_raw", Image, Image_to_opencv, queue_size=1)

    rate=rospy.Rate(10)
    rate.sleep()
    
    rospy.spin()

