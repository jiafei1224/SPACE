import cv2
import os
import argparse
import yaml
import ast
import json


def convert_avi_to_frame(video_path, frame_path):
    """
    Get frames from .avi video files.
    """
    os.makedirs(frame_path, exist_ok=True)

    videos = [os.path.join(video_path, i) for i in os.listdir(video_path)]

    for _, video in enumerate(videos):
        video_id = video.split('/')[-1].split('.')[0]
        video_folder = os.path.join(frame_path, video_id)
        os.makedirs(video_folder, exist_ok=True)
        video_cap = cv2.VideoCapture(video)
        success, image = video_cap.read()
        count = 0
        while success:
            # save frame as JPEG file
            cv2.imwrite(os.path.join(video_folder, "frame_{}.jpg".format(count)), image)    
            success, image = video_cap.read()
            count += 1


ufc_folder_path = ''
save_frame_path = ''

for video_path in os.listdir(ufc_folder_path):

    video_path = os.path.join(ufc_folder_path, video_path)
 
    print("Getting frames from {} and saving to {}...".format(video_path, save_frame_path))
    convert_avi_to_frame(video_path, save_frame_path)
    print("Done.")
    
