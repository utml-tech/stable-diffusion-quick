import cv2
import base64
from PIL import Image

import cv2
import base64
from PIL import Image
from io import BytesIO

from rich.progress import track

import numpy as np

def video_to_frames(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video file is successfully opened
    if not video.isOpened():
        print("Error opening video file")
        return

    frames = []
    frame_count = 0

    # Read until the end of the video
    while video.isOpened():
        # Read the next frame from the video
        ret, frame = video.read()

        if not ret:
            break

        # Convert the frame from BGR to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a PIL Image from the RGB frame
        pil_image = Image.fromarray(rgb_frame)

        # Convert the PIL Image to base64 string
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode("ascii")
        
        # Add the encoded image to the frames list
        frames.append(encoded_image)

        frame_count += 1

    # Release the video file and return the frames
    video.release()
    return frames

def get_video_fps(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video file is successfully opened
    if not video.isOpened():
        print("Error opening video file")
        return None

    # Retrieve the fps value
    fps = video.get(cv2.CAP_PROP_FPS)

    # Release the video file
    video.release()

    return fps

def frames_to_video(frames, output_path, fps=30):
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, get_frame_size(frames[0]))

    # Iterate over the frames
    for frame_data in track(frames):
        # Convert the base64 string to PIL Image
        image_data = base64.b64decode(frame_data)
        pil_image = Image.open(BytesIO(image_data))

        # Convert the PIL Image to numpy array
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Write the frame to the video file
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()


def decode_base64_image(data: str):
    image_data = base64.b64decode(data)
    return Image.open(BytesIO(image_data))

def get_frame_size(frame_data):
    return decode_base64_image(frame_data).size
