from typing import TypedDict
class Keypoint(TypedDict):
    x: int
    y: int
    id: str
    idx: int


from PIL import Image, ImageDraw, ImageFont

def get_text_frame(size, text):
    img = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(img)
    

    draw.text((5, 5), text, fill='black')
    
    return img

import numpy as np
from PIL import Image
from typing import List

def frames_to_np(frames: List[Image.Image]) -> np.ndarray:
    """
    Convert a list of PIL.Image frames to a 4D NumPy array.

    Parameters:
    frames (List[PIL.Image]): List of image frames

    Returns:
    np.ndarray: 4D NumPy array of shape (num_frames, height, width, 3)
    """
    # Initialize an empty list to store NumPy arrays
    array_list = []

    # Loop through each frame
    for frame in frames:
        # Convert PIL image to NumPy array and append to the list
        array_list.append(np.array(frame))

    # Stack arrays along a new axis
    return np.stack(array_list)
