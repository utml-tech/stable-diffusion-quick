import math
from typing import List
from dataclasses import dataclass
import numpy as np
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from tempfile import NamedTemporaryFile
import more_itertools

@dataclass
class VideoProcessor:
    input_path: str
    output_path: str
    batch_size: int = 16

    def __post_init__(self):
        self.clip = None
        self.writer = None
        self.current_frame = 0
        self.audio_file = NamedTemporaryFile(suffix=".mp3", delete=True)

    def __enter__(self):
        self.clip = VideoFileClip(self.input_path)
        if self.clip.audio:  # If the original video has audio
            # Save the original audio to a temporary mp3 file
            self.clip.audio.write_audiofile(self.audio_file.name)
        self.writer = FFMPEG_VideoWriter(self.output_path, self.clip.size, self.clip.fps, audiofile=self.audio_file.name)
        return self

    def __exit__(self, type, value, traceback):
        if self.writer is not None:
            self.writer.close()

        if self.clip is not None:
            self.clip.close()

    def __len__(self):
        return math.ceil(self.clip.duration * self.clip.fps / self.batch_size)

    def __iter__(self):
        yield from more_itertools.batched(self.clip.iter_frames(), self.batch_size)

    def write(self, frames: List[np.ndarray]):
        for frame in frames:
            self.writer.write_frame(frame)
