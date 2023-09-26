
import os

VIDEOS_DIR = '/export/home/mandreev/midres'

video_paths = [os.path.join(VIDEOS_DIR, f) for f in os.listdir(VIDEOS_DIR) if f.endswith('.mp4')]
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
examples = [
    [video_paths[i], video_path_to_prompt_map[get_basename(video_paths[i])], 1, 261] for i in range(len(video_paths))
]

DEFAULT_VIDEO_PATH = os.path.join(VIDEOS_DIR, 'bmx_short.mp4')
