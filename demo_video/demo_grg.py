import os
import time

import cv2
import numpy as np
from demo_utils import render_BallEnv, load_data

from IPython.display import Image


## hyperparameters

# data_dir = 'data'
data_dir = './data/data_Ball_v0/demo'

time_step = 500
data_names = ['attr', 'state', 'action', 'rel_attr']

## Two balls, no relation

# file_name = '2_balls_wo_relation'
file_name = '1'
data_path = '%s/%s.h5' % (data_dir, file_name)

attr, state, action, rel_attr = load_data(data_names, data_path)

render_BallEnv(state, action, rel_attr[0], video=True, image=True, path='%s/%s' % (data_dir, file_name))

print("Sample image:")
sample_img_path = '%s/%s/fig_0.png' % (data_dir, file_name)
Image(filename=sample_img_path) 


## Two balls, one relation

file_name = '2_balls_w_relation'
data_path = '%s/%s.h5' % (data_dir, file_name)

attr, state, action, rel_attr = load_data(data_names, data_path)

render_BallEnv(state, action, rel_attr[0], video=True, image=True,
               path='%s/%s' % (data_dir, file_name),
               draw_edge=False)

render_BallEnv(state, action, rel_attr[0], video=True, image=True,
               path='%s/%s_visEdge' % (data_dir, file_name),
               draw_edge=True)

print("Sample image:")
sample_img_path = '%s/%s_visEdge/fig_0.png' % (data_dir, file_name)
Image(filename=sample_img_path) 