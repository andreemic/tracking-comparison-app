# requires omnimotion directory to be in sys.path

import gradio as gr
from .. import Tracker
import os
import sys
import numpy as np
from src.utils import Keypoint, get_text_frame
import torch
import gc
from src.trackers.dift.dift_video.src.utils import save_frames_as_video, draw_keypoints_on_frames, load_video_as_frames

OMNIMOTION_DIR = '/export/home/mandreev/omnimotion'
sys.path.append(OMNIMOTION_DIR)



from src.examples import get_path


class OmniMotionTracker(Tracker):
    def __init__(self):
        self._name = "OmniMotion"
        self._description = ""
        
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    def get_custom_inputs(self):
        return []
        
    def track(self, frames, existing_video_basename, keypoints: list[Keypoint], source_frame_idx, custom_inputs, device="cuda:0") -> str:
        """
        Args:
            - frames `list[PIL.Image.Image]`: list of frames (ignored here since video is already tracked)
            - existing_video_basename `str`: basename of the video that we're tracking; used to identify the previously tracked omnimotion video
            - keypoints `list[{x, y, id, idx}]`: list of keypoints to track
            - source_frame_idx `int`: index of the source frame
            - custom_inputs `list`: list of custom inputs used as configuration for the omnimotion inference procedure
            - device `str`: device to use for inference
        Returns:
            - `list[PIL.Image.Image]`: list of frames with keypoints drawn on them
        """
        print(f'OmniMotion started with {len(keypoints)} keypoints and {len(frames)} frames')

        if not existing_video_basename or not os.path.exists(get_path(existing_video_basename)):
            print(f'Skipping OmniMotion tracker because no video basename was provided')
            return [get_text_frame(frames[0].size, "No OmniMotion model found for this video.")]
        try:

            from config import config_parser
            from trainer import BaseTrainer

            if not existing_video_basename or not os.path.exists(get_path(existing_video_basename)):
                print('no omnimotion model for this video, skipping')
                return frames

            else:
                print(f'⌛ Tracking keypoints with OmniMotion for video {existing_video_basename}...')
                
            if len(keypoints) == 0:
                print('no keypoints to track, skipping')
                return frames

            torch.manual_seed(1234)
            args = config_parser(True)
            args.expname = 'default'
            args.data_dir = os.path.join(OMNIMOTION_DIR, 'omnimotion_videos', existing_video_basename)
            args.save_dir = os.path.join(OMNIMOTION_DIR, 'out')
            args.query_frame_id = source_frame_idx
            args.load_opt = 1
            args.use_affine = True
            args.use_max_loc = True
            args.vis_occlusion = True
            args.use_error_map = True
            args.use_count_map = True
            
            args.config = os.path.join(OMNIMOTION_DIR, 'configs', 'default.txt')

            if not os.path.exists(args.data_dir):
                raise ValueError(f'No omnimotion video data found at {args.data_dir}')
            
            trainer = BaseTrainer(args)

            # we need to transform keypoints into (len(keypoints), 2) tensor of xy coordinates that are relative to omnimotion frame sizes
            def map_x_to_omnimotion_frame_size(x):
                return x * trainer.w / frames[0].size[0]
            def map_y_to_omnimotion_frame_size(y):
                return y * trainer.h / frames[0].size[1]
            prepped_keypoints = np.array([[map_x_to_omnimotion_frame_size(kp['x']), map_y_to_omnimotion_frame_size(kp['y'])] for kp in keypoints])

            print(f'OmniMotion: prepped keypoints: {prepped_keypoints}')
            print(f'OmniMotion: args: {args}')
            # tracked_kpts has shape (len(frames), number_keypoints, 3), where 3 is (x, y, confidence)
            omnimotion_frames, tracked_kpts = trainer.eval_video_correspondences(source_frame_idx, 
                pts=prepped_keypoints,
                vis_occlusion=True,
                occlusion_th=0.99,
                use_max_loc=True,
                radius=3,
                return_kpts=True, verbose=True, side_by_side=False)

            print(f'OmniMotion: tracked_kpts: {tracked_kpts}')

            del trainer, BaseTrainer, config_parser
            gc.collect()
            torch.cuda.empty_cache()

            return omnimotion_frames

            # convert tracked_kpts to list of keypoints
            # keypoints_per_frame = []
            # for frame_kpts in tracked_kpts:
            #     frame_keypoints = []
            #     for i, kp in enumerate(frame_kpts):
            #         frame_keypoints.append({
            #             **keypoints[i],
            #             'x': kp[0].item(),
            #             'y': kp[1].item(),
            #         })
            #     keypoints_per_frame.append(frame_keypoints)
            
            # # draw keypoints on frames
            # frames_with_dots = draw_keypoints_on_frames(frames, keypoints_per_frame)
            
            # return frames_with_dots
        except Exception as e:
            print(f'OmniMotion: exception: {e}')
            return frames

