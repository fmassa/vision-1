import os
import torch

from common_utils import get_tmp_dir
import unittest
from torchvision import io

import contextlib

from torchvision.datasets.video_utils import VideoClips



@contextlib.contextmanager
def get_list_of_videos(num_videos=5):
    with get_tmp_dir() as tmp_dir:
        names = []
        for i in range(num_videos):
            data = torch.randint(0, 255, (5 * (i + 1), 300, 400, 3), dtype=torch.uint8)
            name = os.path.join(tmp_dir, "{}.mp4".format(i))
            names.append(name)
            io.write_video(name, data, fps=5)

        yield names


class Tester(unittest.TestCase):
    def test_video_clips(self):
        with get_list_of_videos(num_videos=3) as video_list:
            video_clips = VideoClips(video_list, 5, 5)
            self.assertEqual(video_clips.num_clips(), 1 + 2 + 3)

            video_clips = VideoClips(video_list, 6, 6)
            self.assertEqual(video_clips.num_clips(), 0 + 1 + 2)

            video_clips = VideoClips(video_list, 6, 1)
            self.assertEqual(video_clips.num_clips(), 0 + (10 - 6 + 1) + (15 - 6 + 1))


if __name__ == '__main__':
    unittest.main()
