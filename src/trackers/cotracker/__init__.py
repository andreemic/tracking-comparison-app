import os
import torch

from src.utils import frames_to_np, Keypoint
from typing import List
import sys
sys.path.append("./src/trackers/cotracker/co-tracker")
from cotracker.utils.visualizer import Visualizer, read_video_from_path

import gradio as gr
from . import Tracker

from gpu_utils import get_suitable_device
from src.trackers.dift.dift_video.src.utils import save_frames_as_video, draw_keypoints_on_frames, load_video_as_frames
from performance import PerformanceManager

import math
import time
from PIL import Image

def current_milli_time():
    return str(round(time.time() * 1000))
def keypoints_to_queries(keypoints: List[Keypoint], source_frame_idx: int):
    # keypoints: List[{x: int, y: int, id: int, idx: int}]
    # convert to 
    # queries: Queried points of shape (1, N, 3) in format (t, x, y) for frame index and pixel coordinates.
    queries = []
    for keypoint in keypoints:
        queries.append([source_frame_idx, keypoint['x'], keypoint['y']])

    return torch.tensor(queries).unsqueeze(0).float()


class CoTrackerTracker(Tracker):
    def __init__(self):
        self._name = "Co-Tracker"
        self._description = "https://github.com/facebookresearch/co-tracker/tree/main"
        
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    def get_custom_inputs(self):
        return [gr.Checkbox(label="Backward Tracking", default=True)]
        
    def track(self, frames, existing_video_basename, keypoints: List[Keypoint], source_frame_idx, custom_inputs, device="cuda:0") -> str:
        backward_tracking = custom_inputs[0]

        frames_np = frames_to_np(frames)
        source_frame_idx = min(len(frames_np)-1, source_frame_idx)
        frames_pt = torch.from_numpy(frames_np).permute(0, 3, 1, 2)[None].float().to(device)


        model = torch.hub.load("facebookresearch/co-tracker", "cotracker_w8").to(device)
        
        queries = keypoints_to_queries(keypoints, source_frame_idx).to(device)
        
        pred_tracks, pred_visibility = model(
            frames_pt, 
            queries=queries, 
            backward_tracking=backward_tracking
        )

        linewidth = 2
            
        tracks_leave_trace = False
        temp_save_dir = os.path.join(os.path.dirname(__file__), "cotracker_temp")
        vis = Visualizer(
            save_dir=temp_save_dir,
            grayscale=False,
            pad_value=100,
            fps=10,
            linewidth=linewidth,
            show_first_frame=5,
            tracks_leave_trace= -1 if tracks_leave_trace else 0,
        )


        filename = current_milli_time()

        # tracked_frames: [N, frames, 3, height, width])
        tracked_frames = vis.visualize(
            frames_pt,
            tracks=pred_tracks, 
            visibility=pred_visibility,
            query_frame=source_frame_idx,
            save_video = True,
            filename=filename,
        )

        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return tracked_frames_to_pil(tracked_frames)


def tracked_frames_to_pil(tracked_frames):
    """
    Converts [1, frames, 3, height, width] tensor to array of PIL images
    """

    assert len(tracked_frames.shape) == 5, f"Expected 5D tensor, got {len(tracked_frames.shape)}D tensor"
    assert tracked_frames.shape[0] == 1, f"Expected first dimension to be 1, got {tracked_frames.shape[0]}"

    tracked_frames = tracked_frames.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
    tracked_frames = [Image.fromarray(tracked_frame) for tracked_frame in tracked_frames]
    return tracked_frames