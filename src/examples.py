
import os

VIDEOS_DIR = '/export/home/mandreev/midres'

OMNIMOTION_RESULT_DIR = '/export/home/mandreev/omnimotion/out'
# Warning: requires for each subdirectory in omnimotion_videos to exist a same-named .mp4 file in VIDEOS_DIR

# todo:
# - add a function to each tracker class called "processed_video_ids"
# - build a unifying structure in app.py for selecting from them and skipping trackers if selected video is not processed by this tracker
# - move this omnimotion-specific logic into trackers/omnimotion
video_paths = [os.path.join(VIDEOS_DIR, f + '.mp4') for f in os.listdir('/export/home/mandreev/omnimotion/omnimotion_videos')]


video_path_to_prompt_map = {
    'bmx_short': 'a bmx biker in a skatepark',
    'cat_grass': 'a cat in grass',
    'cat_walk': 'a cat on the floor outside',
    'dancers_short': 'dancers performing a dance inside a room',
    'monkey_short': 'a small monkey eating a banana',
    'soccer_short': 'a soccer player kicking a ball',
    'violin': 'a violinist playing a violin outside in a jacket',
    'walking': 'a crowd of people walking in a city',
    'walking_2': 'a crowd of people walking in a city',
    'dogs': 'two dogs playing in the grass',
    'dogs_2': 'two dogs playing in the snow',
}
def get_basename(fpath):
    return os.path.basename(fpath).split('.')[0]

def get_path(basename):
    return os.path.join(VIDEOS_DIR, basename + '.mp4')

basenames = [get_basename(fpath) for fpath in video_paths if os.path.exists(os.path.join(OMNIMOTION_RESULT_DIR, f'default_{get_basename(fpath)}'))]
examples = [
    [video_paths[i], video_path_to_prompt_map[get_basename(video_paths[i])], 1, 261] for i in range(len(video_paths))
]

DEFAULT_VIDEO_PATH = os.path.join(VIDEOS_DIR, 'bmx_short.mp4')
