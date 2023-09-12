import os
DIFT_DIR = '/export/home/mandreev/dift/src/models'
DIFT_VIDEO_DIR = '/export/home/mandreev/dift-video/src'
OMNIMOTION_DIR = '/export/home/mandreev/omnimotion'
VIDEOS_DIR = '/export/home/mandreev/midres'


import sys
sys.path.append(DIFT_DIR)
sys.path.append(DIFT_VIDEO_DIR)
sys.path.append(OMNIMOTION_DIR)

from keypoint_extractor import KeypointExtractor
from gpu_utils import get_suitable_device

from utils import load_video_as_frames, draw_keypoints_on_frames, save_frames_as_video

video_paths = [os.path.join(VIDEOS_DIR, f) for f in os.listdir(VIDEOS_DIR) if f.endswith('.mp4')]
video_path_to_prompt_map = {
    'bmx_short': 'a bmx biker in a skatepark',
    'cat_grass': 'a cat in grass',
    'cat_walk': 'a cat on the floor outside',
    'dancers_short': 'dancers performing a dance inside a room',
    'monkey_short': 'a small monkey eating a banana',
    'soccer_short': 'a soccer player kicking a ball',
    'violin': 'a violinist playing a violin outside in a jacket'
}
def get_basename(fpath):
    return os.path.basename(fpath).split('.')[0]
examples = [
    [video_paths[i], video_path_to_prompt_map[get_basename(video_paths[i])], 1, 261] for i in range(len(video_paths))
]

DEFAULT_VIDEO_PATH = os.path.join(VIDEOS_DIR, 'bmx_short.mp4')


import numpy as np
import gradio as gr



dift_extractor = None
def dift_track(video_path, prompt, layer=1, step=261,progress=gr.Progress()):

    global dift_extractor
    if not dift_extractor:
        dift_extractor = KeypointExtractor(device='cuda:1')

    progress(0.1, desc='Tracking keypoints for each frame')
    print(f'⌛ Tracking keypoints for {video_path}...')
    frames = load_video_as_frames(video_path)
    keypoints = dift_extractor.track_keypoints(frames[:10], prompt=prompt, source_frame_idx=0, grid_size=10)
    print(f'✅ Done tracking keypoints for {video_path}.')

    progress(0.5, desc='Drawing keypoints on frames')
    print(f'⌛ Drawing keypoints on frames for {video_path}...')
    frames_with_dots = draw_keypoints_on_frames(frames, keypoints)
    print(f'✅ Done drawing keypoints on frames for {video_path}.')

    progress(0.8, desc='Saving frames with keypoints to video')
    print(f'⌛ Saving frames with keypoints to video for {video_path}...')
    video_path = f'{video_path.split(".")[0]}_tracked.mp4'
    save_frames_as_video(frames_with_dots, video_path)
    print(f'✅ Done saving frames with keypoints to video for {video_path}.')
    
    progress(1.0, desc='Done!')
    print(video_path)
    return video_path

video_input = gr.Video(default=DEFAULT_VIDEO_PATH, label="Video")
prompt_input = gr.Textbox(placeholder="Enter a prompt here...", default="a bmx biker in a skatepark")
layer_input = gr.Number(default=1, label="UNet Layer")
step_input = gr.Number(default=261, label="Timestep")

demo = gr.Interface(dift_track, [video_input, prompt_input, layer_input, step_input], "playable_video", examples=examples)

demo.queue().launch()