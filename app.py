# required paths 
import sys
sys.path.append('./src')
sys.path.append("./src/trackers/dift/dift_video/src")

from examples import examples, DEFAULT_VIDEO_PATH

# packages
import numpy as np
import gradio as gr
import math
from PIL import Image
from common_inputs import pick_source_frame, pick_keypoints, pick_video_components,generate_keypoints_from_mask
import concurrent.futures
import traceback

import sys


from src.trackers.dift.dift_video.src.utils import save_frames_as_video, load_video_as_frames

# tracker implementations
from trackers.dift import DIFTTracker
from trackers.omnimotion import OmniMotionTracker
from trackers.cotracker import CoTrackerTracker
trackers = [
    DIFTTracker(),
    OmniMotionTracker(),
    CoTrackerTracker()
]
device = 'cuda:0'

tracker_custom_inputs_counts = {}

import concurrent.futures
import math
import traceback

PARALLEL_INFERENCE = False
# tracks a video with all trackers

def track(video_path, existing_video_basename, keypoints, source_frame_percentage, from_frame_percentage, to_frame_percentage, source_time_input, from_frame, to_frame, picked_frame, *custom_inputs) -> list:
    """
        Converts video path to frames array, calls trackers, writes results back to video files, returns list of video paths.
        So basically: takes care of Gradio-specific IO.
    """
    print(video_path, keypoints, source_frame_percentage, custom_inputs)
    
    _tracker_outputs = []
    custom_inputs_offset = 0
    
    frames = load_video_as_frames(video_path, max_dimension=640)
    source_frame_idx = math.floor(len(frames) * source_frame_percentage)
    start_frame_idx = math.floor(len(frames) * from_frame_percentage)
    end_frame_idx = math.floor(len(frames) * to_frame_percentage)

    frames = frames[start_frame_idx:end_frame_idx]
    
    if existing_video_basename is None:
        existing_video_basename = video_path.split('/')[-1].split('.')[0]

    def track_and_save(i, tracker):
        custom_inputs_count = tracker_custom_inputs_counts[tracker.name]
        tracker_custom_inputs = custom_inputs[custom_inputs_offset:custom_inputs_offset+custom_inputs_count]
        frames_with_keypoints = tracker.track(frames, existing_video_basename, keypoints, source_frame_idx, tracker_custom_inputs, device=device)
        out_video_path = f'{video_path.split(".")[0]}_tracked_{i}.webm'
        save_frames_as_video(frames_with_keypoints, out_video_path)
        return out_video_path

    if PARALLEL_INFERENCE:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_output = {executor.submit(track_and_save, i, tracker): i for i, tracker in enumerate(trackers)}
            for future in concurrent.futures.as_completed(future_to_output):
                i = future_to_output[future]
                try:
                    out_video_path = future.result()
                    _tracker_outputs.append({
                        "tracker_name": trackers[i].name,
                        "out_video_path": out_video_path
                    })
                except Exception as e:
                    _tracker_outputs.append({
                        "tracker_name": trackers[i].name,
                        "out_video_path": None
                    })
                    print(f'Tracker {i} generated an exception: {str(e)}')
                    traceback.print_exc()
    else:
        for i, tracker in enumerate(trackers):
            try:
                out_video_path = track_and_save(i, tracker)
                _tracker_outputs.append({
                    "tracker_name": tracker.name,
                    "out_video_path": out_video_path
                })
            except Exception as e:
                _tracker_outputs.append({
                    "tracker_name": tracker.name,
                    "out_video_path": None
                })
                print(f'Tracker {i} generated an exception: {str(e)}')
                traceback.print_exc()

    tracker_names = [tracker.name for tracker in trackers]
    _tracker_outputs = sorted(_tracker_outputs, key=lambda x: tracker_names.index(x['tracker_name']))
    
    tracker_output = list(map(lambda x: x['out_video_path'], _tracker_outputs))

    print(f'Returning tracker output: {tracker_output}')
    return [*tracker_output, source_time_input, from_frame, to_frame, picked_frame]

# UI
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    keypoints = gr.State(value=[])
    picked_frame_clean = gr.State(value=None)
    with gr.Row():
        with gr.Column():
            existing_video_select, video_input = pick_video_components()
            picked_frame = gr.Image(label="Pick keypoints", interactive=False)
            picked_frame_mask = gr.Image(label="Draw Mask", interactive=True, tool='sketch', brush_size=20)

            # when a source time is inputted, populate the picked frame with the frame at that time
            with gr.Group():
                source_time_input = gr.Slider(minimum=0, default=0, maximum=1, label="Pick source frame", interactive=True)
                source_time_input.release(pick_source_frame, [video_input,source_time_input], [picked_frame, keypoints, picked_frame_clean])
                with gr.Row():
                    from_frame = gr.Number(value=0, label="From", visible=False)
                    to_frame = gr.Number(value=1, label="To", visible=False)
                
            
            # when a video is uploaded, populate the picked frame with the first frame
            video_input.change(pick_source_frame, [video_input, source_time_input], [picked_frame, picked_frame_mask, keypoints, picked_frame_clean])

            # when the frame is clicked, pick the keypoint
            picked_frame.select(pick_keypoints, [picked_frame, keypoints], [picked_frame, keypoints])

            # when a mask is drawn, convert it to keypoints
            picked_frame_mask.edit(generate_keypoints_from_mask, [picked_frame, picked_frame_mask, keypoints, picked_frame_clean], [picked_frame, keypoints])

            tracker_inputs_list = []
            
            for tracker in trackers:
                with gr.Accordion(f"{tracker.name} Inputs", open=False):
                    custom_inputs = tracker.get_custom_inputs()
                    tracker_inputs_list.extend(custom_inputs)
                    tracker_custom_inputs_counts[tracker.name] = len(custom_inputs)
            
        with gr.Column():
            tracker_outputs = [gr.Video(label=f"{tracker.name} Output") for tracker in trackers]
            generate_button = gr.Button(label="Generate")

            generate_button.click(track, [video_input, existing_video_select, keypoints, source_time_input, from_frame, to_frame, source_time_input, from_frame, to_frame, picked_frame, *tracker_inputs_list], [*tracker_outputs, source_time_input, from_frame, to_frame, picked_frame])



    # # gr.Markdown("## Video Examples")
    # gr.Examples(examples, label="Examples", inputs=[video_input, prompt_input, layer_input, step_input])


if __name__ == '__main__':
    demo.queue()
    demo.launch(share=True)
