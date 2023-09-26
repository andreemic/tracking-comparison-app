from utils import load_video_as_frames, load_video_as_frames
from PIL import Image
import gradio as gr
import math
from utils import draw_keypoint_on_frame

def pick_source_frame(video_path, source_time_percentage):
    frames = load_video_as_frames(video_path, fps=24)

    duration_s = len(frames) / 24

    source_frame_idx = math.floor(len(frames) * source_time_percentage)

    if source_frame_idx >= len(frames):
        raise ValueError(f"Source time {source_time_s} is greater than video duration {duration_s}s")
    
    frame = frames[source_frame_idx]
    
    print(frame)
    return [frame, []]

def get_new_keypoint_id(keypoints):
    if not keypoints or len(keypoints) == 0:
        return 'kp_0'
    last_kp = keypoints[-1]
    last_strint_id = last_kp["kp_id"].split("_")[-1]
    return f'kp_{int(last_strint_id) + 1}'

def pick_keypoints(source_frame_img, keypoints, evt: gr.SelectData):
    """ 
    Pick keypoints from a frame image.
    """
    source_frame_img = Image.fromarray(source_frame_img)
    x, y = tuple(evt.index)
    print(f'Picked keypoint at ({x}, {y})')

    new_keypoint_dict = {
        'x': x,
        'y': y,
        'kp_id': get_new_keypoint_id(keypoints),
        'idx': len(keypoints),
    }
    keypoints.append(new_keypoint_dict)

    frame_with_keypoint = source_frame_img.copy()
    frame_with_keypoint = draw_keypoint_on_frame(frame_with_keypoint, new_keypoint_dict, style='cross', color="auto", radius=6)

    return [frame_with_keypoint, keypoints]