
import sys
import argparse
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2

import lib.transform_cv2 as T
from lib.models import model_factory
from configs import set_cfg_from_file
import types
import time


torch.set_grad_enabled(False)
np.random.seed(123)


args = types.SimpleNamespace()
args.config             = 'configs/bisenetv2_coco.py'
args.weight_path        = '/home/cordin/BiSeNet/model_final_v2_coco.pth'


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



capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


while cv2.waitKey(33) < 0:
    ret, frame = capture.read()
    print(ret)
    start_time = time.time()
    im = frame[:, :, ::-1]

    im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

    # inference
    out = net(im).squeeze().detach().cpu().numpy()
    pred = palette[out]
    cv2.imshow("VideoFrame", pred)
    cv2.imshow("raw", frame)

    elapsed_time = time.time() - start_time
    print('Elapesed time: %s' % str(elapsed_time))

capture.release()
cv2.destroyAllWindows()
