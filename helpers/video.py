from __future__ import annotations

import math
from typing import Any, Iterator, List, Optional, Tuple
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

    def __enter__(self) -> VideoProcessor:
        self.audio_file = NamedTemporaryFile(suffix=".mp3", delete=True)
        self.current_frame = 0

        self.clip = VideoFileClip(self.input_path)
        self.writer = FFMPEG_VideoWriter(self.output_path, self.clip.size, self.clip.fps, audiofile=self.write_audiofile(self.clip, self.audio_file.name))
        return self

    def __exit__(self, type: type | None, value: Exception | None, traceback: Any) -> None:
        if self.writer is not None:
            self.writer.close()

        if self.clip is not None:
            self.clip.close()

    def __len__(self) -> int:
        return math.ceil(self.clip.duration * self.clip.fps / self.batch_size)

    def __iter__(self) -> Iterator[list[np.ndarray]]:
        yield from more_itertools.batched(self.clip.iter_frames(), self.batch_size)

    def write(self, frames: np.ndarray) -> None:
        for frame in frames:
            self.writer.write_frame(frame)

    @staticmethod
    def write_audiofile(clip: VideoFileClip, filepath: str) -> str:
        if clip.audio:  # If the original video has audio
            # Save the original audio to a temporary mp3 file
            clip.audio.write_audiofile(filepath)
        return filepath
