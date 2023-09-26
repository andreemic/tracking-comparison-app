# required paths 
DIFT_DIR = '/export/home/mandreev/dift/src/models'
DIFT_VIDEO_DIR = '/export/home/mandreev/dift-video/src'
OMNIMOTION_DIR = '/export/home/mandreev/omnimotion'

import sys
sys.path.append(DIFT_DIR)
sys.path.append(DIFT_VIDEO_DIR)
sys.path.append(OMNIMOTION_DIR)
from src.examples import examples, DEFAULT_VIDEO_PATH

# packages
import numpy as np
import gradio as gr
import math
from PIL import Image
from src.common_inputs import pick_source_frame, pick_keypoints

# tracker implementations
from trackers.dift import DIFTTracker
from trackers.omnimotion import OmniMotionTracker
trackers = [
    DIFTTracker(),
    OmniMotionTracker()
]

tracker_custom_inputs_counts = {}

# tracks a video with all trackers
def track(video_path, keypoints, source_frame_percentage, *custom_inputs) -> list:
    print(video_path, keypoints, source_frame_percentage, custom_inputs)
    
    _tracker_outputs = []
    custom_inputs_offset = 0
    for i, tracker in enumerate(trackers):
        custom_inputs_count = tracker_custom_inputs_counts[tracker.name]
        tracker_custom_inputs = custom_inputs[custom_inputs_offset:custom_inputs_offset+custom_inputs_count]
        custom_inputs_offset += custom_inputs_count
        tracker_output = tracker.track(video_path, keypoints, source_frame_percentage, tracker_custom_inputs)
        # tracker_output = '/export/home/mandreev/dift-video/test.webm'
        _tracker_outputs.append(tracker_output)

    return _tracker_outputs

# UI
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    keypoints = gr.State(value=[])
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(value=examples[0][0], label="Video")
            picked_frame = gr.Image(label="Source Frame", interactive=False)

            # when a source time is inputted, populate the picked frame with the frame at that time
            source_time_input = gr.Slider(minimum=0, default=0, maximum=1, label="Pick source frame", interactive=True)
            source_time_input.release(pick_source_frame, [video_input,source_time_input], [picked_frame, keypoints])
            
            # when a video is uploaded, populate the picked frame with the first frame
            video_input.change(pick_source_frame, [video_input, source_time_input], [picked_frame, keypoints])

            # when the frame is clicked, pick the keypoint
            picked_frame.select(pick_keypoints, [picked_frame, keypoints], [picked_frame, keypoints])

            tracker_inputs_list = []
            
            for tracker in trackers:
                with gr.Accordion(f"{tracker.name} Inputs"):
                    custom_inputs = tracker.get_custom_inputs()
                    tracker_inputs_list.extend(custom_inputs)
                    tracker_custom_inputs_counts[tracker.name] = len(custom_inputs)
            
        with gr.Column():
            tracker_outputs = [gr.Video(label=f"{tracker.name} Output") for tracker in trackers]
            generate_button = gr.Button(label="Generate")


            generate_button.click(track, [video_input, keypoints, source_time_input, *tracker_inputs_list], tracker_outputs)



    # # gr.Markdown("## Video Examples")
    # gr.Examples(examples, label="Examples", inputs=[video_input, prompt_input, layer_input, step_input])

demo.launch()