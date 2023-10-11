from utils import load_video_as_frames, load_video_as_frames
from PIL import Image
import gradio as gr
import math
from utils import draw_keypoint_on_frame

def pick_source_frame(video_path, source_time_percentage):
    print(f'[pick_source_frame] video_path={video_path}, source_time_percentage={source_time_percentage}')
    if video_path is None:
        return [None, None, [], None]
    frames = load_video_as_frames(video_path, fps=24, max_dimension=640)

    if len(frames) == 0:
        return [None, None, [], None]

    duration_s = len(frames) / 24

    source_frame_idx = math.floor(len(frames) * source_time_percentage)

    if source_frame_idx >= len(frames):
        raise ValueError(f"Source time {source_time_percentage} is greater than video duration {duration_s}s")
    
    frame = frames[source_frame_idx]
    
    print(f'[pick_source_frame] picked frame {source_frame_idx} of {len(frames)}')
    return [frame, frame, [], frame]


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
    

    frame_with_keypoint = source_frame_img.copy()
    frame_with_keypoint, keypoint_color = draw_keypoint_on_frame(frame_with_keypoint, new_keypoint_dict, style='cross', color="auto", radius=6)

    new_keypoint_dict['color'] = keypoint_color
    keypoints.append(new_keypoint_dict)

    return [frame_with_keypoint, keypoints]


# how many points a grid should have across the entire frame
GRID_WIDTH = 30
GRID_HEIGHT = 30
def generate_keypoints_from_mask(picked_frame, picked_frame_mask, keypoints, picked_frame_clean, **args):
    print(picked_frame_mask['mask'].shape,  args)

    h, w, _ = picked_frame.shape

    # mask is np.array of shape (height, width, 3) with values in [0, 255]
    # pick keypoints that are within the mask
    mask = picked_frame_mask['mask']
    mask = mask[:, :, 0]
    mask = mask / 255
    
    keypoints = []
    frame_with_keypoints = Image.fromarray(picked_frame).copy()
    for i in range(GRID_WIDTH):
        for j in range(GRID_HEIGHT):
            x = math.floor(i * w / GRID_WIDTH)
            y = math.floor(j * h / GRID_HEIGHT)
            if mask[y][x] == 1:
                new_keypoint_dict = {
                    'x': x,
                    'y': y,
                    'kp_id': get_new_keypoint_id(keypoints),
                    'idx': len(keypoints),
                }
                keypoints.append(new_keypoint_dict)
                frame_with_keypoints, keypoint_color = draw_keypoint_on_frame(frame_with_keypoints, new_keypoint_dict, style='cross', color="auto", radius=6)
    if len(keypoints) == 0:
        return [picked_frame_clean, keypoints]

    return [frame_with_keypoints, keypoints]

from src.examples import basenames, examples, get_path, video_paths
def pick_video_components():
    def select_existing_video(existing_video_select, video_input, evt: gr.SelectData):
        if existing_video_select is None:
            return [existing_video_select, video_input]

        video_path = get_path(existing_video_select)
        return [existing_video_select, video_path]

    with gr.Group():
        existing_video_select = gr.Dropdown(options=basenames,
            value=basenames[0],
            choices=basenames,
            max_choices=1,
            label="Choose existing video", placeholder="Choose a video")
        with gr.Accordion("Watch or upload new video", open=False):
            video_input = gr.Video(value=video_paths[0], label="Input video")

            # video_input.upload(
            #     fn=lambda video_input: None,
            #     inputs=[video_input],
            #     outputs=[existing_video_select]
            # )
            video_input.clear(
                fn=lambda video_input: None,
                inputs=[video_input],
                outputs=[existing_video_select]
            )

        existing_video_select.select(fn=select_existing_video, inputs=[existing_video_select, video_input], outputs=[existing_video_select, video_input])


    return existing_video_select, video_input