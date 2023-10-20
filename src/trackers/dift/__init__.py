import gradio as gr
from .. import Tracker

from gpu_utils import get_suitable_device
from .dift_video.src.performance import PerformanceManager

from trackers.dift.dift_video.src.keypoint_extractor import KeypointExtractor
from trackers.dift.dift_video.src.utils import save_frames_as_video, draw_keypoints_on_frames, load_video_as_frames

import math
dift_extractor = None

class DIFTTracker(Tracker):
    def __init__(self, cache_dir='/export/home/mandreev/dift_cache'):
        self._name = "DIFT"
        self._description = "A naive extension of DIFT correspondences to video."
        self.cache_dir = cache_dir
        
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    def get_custom_inputs(self):
        return [gr.Textbox(placeholder="Enter a prompt here..."), gr.Number(label="UNet Layer", value=1), gr.Number(label="Timestep", value=261)]
        
    def track(self, frames, existing_video_basename, keypoints, source_frame_idx, custom_inputs, device="cuda:0") -> str:
        prompt, layer, step = custom_inputs
        global dift_extractor
        if not dift_extractor:
            print(f'⌛ Loading DIFT Extractor...')
            dift_extractor = KeypointExtractor(device=device)

        # progress(0.1, desc='Tracking keypoints for each frame')
        print(f'⌛ Tracking keypoints with DIFT...')
        
        perf_manager = PerformanceManager()
        keypoints = dift_extractor.track_keypoints(
            frames, 
            prompt=prompt, 
            source_frame_idx=source_frame_idx, 
            source_frame_keypoints=keypoints, 
            layer=layer, 
            timestep=step,
            cache_dir=self.cache_dir,
            video_cache_key=existing_video_basename,
            verbose=True,
            perf_manager=perf_manager
            )

        
        # progress(0.5, desc='Drawing keypoints on frames')
        print(f'⌛ DIFT - Drawing keypoints on frames for...')
        perf_manager.start('draw_keypoints_on_frames')
        frames_with_dots = draw_keypoints_on_frames(frames, keypoints)
        perf_manager.end('draw_keypoints_on_frames')
        print(f'✅ DIFT - Done drawing keypoints on frames.')

        perf_manager.print_summary()

        return frames_with_dots
