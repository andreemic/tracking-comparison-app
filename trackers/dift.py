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
        return [gr.Textbox(placeholder="Enter a prompt here..."), gr.Number(label="UNet Layer", value=1), gr.Number(label="Timestep", value=261)]
        
    def track(self, frames, keypoints, source_frame_idx, custom_inputs, device="cuda:0") -> str:
        prompt, layer, step = custom_inputs
        global dift_extractor
        if not dift_extractor:
            print(f'⌛ Loading DIFT Extractor...')
            dift_extractor = KeypointExtractor(device=device)

        # progress(0.1, desc='Tracking keypoints for each frame')
        print(f'⌛ Tracking keypoints with DIFT...')
        
        keypoints = dift_extractor.track_keypoints(
            frames, 
            prompt=prompt, 
            source_frame_idx=source_frame_idx, 
            source_frame_keypoints=keypoints, 
            layer=layer, 
            timestep=step
            )

        # progress(0.5, desc='Drawing keypoints on frames')
        print(f'⌛ DIFT - Drawing keypoints on frames for...')
        frames_with_dots = draw_keypoints_on_frames(frames, keypoints)
        print(f'✅ DIFT - Done drawing keypoints on frames.')

        return frames_with_dots
