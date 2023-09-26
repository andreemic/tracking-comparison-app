import gradio as gr
from . import Tracker

from keypoint_extractor import KeypointExtractor
from gpu_utils import get_suitable_device
from utils import save_frames_as_video, draw_keypoints_on_frames, load_video_as_frames

import math
dift_extractor = None

class DIFTTracker(Tracker):
    def __init__(self):
        self._name = "DIFT"
        self._description = "A naive extension of DIFT correspondences to video."
        
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    def get_custom_inputs(self):
        return [gr.Textbox(placeholder="Enter a prompt here..."), gr.Number(label="UNet Layer"), gr.Number(label="Timestep")]
        
    def track(self, video_path, keypoints, source_frame_percentage, custom_inputs) -> str:
        prompt, layer, step = custom_inputs
        global dift_extractor
        if not dift_extractor:
            print(f'⌛ Loading DIFT Extractor...')
            dift_extractor = KeypointExtractor(device='cuda:1')

        # progress(0.1, desc='Tracking keypoints for each frame')
        print(f'⌛ Tracking keypoints for {video_path}...')
        frames = load_video_as_frames(video_path)
        source_frame_idx = math.floor(len(frames) * source_frame_percentage)
        keypoints = dift_extractor.track_keypoints(
            frames, 
            prompt=prompt, 
            source_frame_idx=source_frame_idx, 
            source_frame_keypoints=keypoints, 
            layer=layer, 
            timestep=step
            )
        print(f'✅ Done tracking keypoints for {video_path}.')

        # progress(0.5, desc='Drawing keypoints on frames')
        print(f'⌛ Drawing keypoints on frames for {video_path}...')
        frames_with_dots = draw_keypoints_on_frames(frames, keypoints)
        print(f'✅ Done drawing keypoints on frames for {video_path}.')

        # progress(0.8, desc='Saving frames with keypoints to video')
        print(f'⌛ Saving frames with keypoints to video for {video_path}...')
        video_path = f'{video_path.split(".")[0]}_tracked.webm'
        save_frames_as_video(frames_with_dots, video_path)
        print(f'✅ Done saving frames with keypoints to video for {video_path}.')
        
        # progress(1.0, desc='Done!')
        print(video_path)
        return video_path
