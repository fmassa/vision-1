import bisect
import torch
from torchvision.io import read_video_timestamps, read_video


def unfold(tensor, size, step, dilation):
    """
    similar to tensor.unfold, but with the dilation
    and specialized for 1d tensors
    """
    assert tensor.dim() == 1
    o_stride = tensor.stride(0)
    numel = tensor.numel()
    new_stride = (step * o_stride, dilation * o_stride)
    new_size = ((numel - (dilation * (size - 1) + 1)) // step + 1, size)
    if new_size[0] < 1:
        new_size = (0, size)
    return torch.as_strided(tensor, new_size, new_stride)


class VideoClips(object):
    def __init__(self, video_paths, clip_length_in_frames=16, frames_between_clips=1):
        self.video_paths = video_paths
        self._compute_frame_pts()
        self.compute_clips(clip_length_in_frames, frames_between_clips)

    def _compute_frame_pts(self):
        self.video_pts = []
        for video_file in self.video_paths:
            clips = read_video_timestamps(video_file)
            self.video_pts.append(torch.as_tensor(clips))

    def compute_clips(self, num_frames, step, dilation=1):
        self.num_frames = num_frames
        self.step = step
        self.dilation = dilation
        self.clips = []
        for video_pts in self.video_pts:
            clips = unfold(video_pts, num_frames, step, dilation)
            self.clips.append(clips)
        l = torch.as_tensor([len(v) for v in self.clips])
        self.cumulative_sizes = l.cumsum(0).tolist()

    def __len__(self):
        return self.num_clips()

    def num_videos(self):
        return len(self.video_paths)

    def num_clips(self):
        return self.cumulative_sizes[-1]

    def get_clip_pts_in_flat_index(self, idx):
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]
        return self.video_paths[video_idx], self.clips[video_idx][clip_idx], video_idx

    def get_all_clip_pts_in_video(self, idx):
        return self.video_paths[idx], self.clips[idx]

    def get_clip(self, idx):
        video_path, clip_pts, video_idx = self.get_clip_pts_in_flat_index(idx)
        video, audio, info = read_video(video_path, clip_pts[0].item(), clip_pts[-1].item())
        video = video[::self.dilation]
        # TODO change video_fps in info?
        assert len(video) == self.num_frames
        return video, audio, info, video_idx


if __name__ == "__main__":
    p = "/datasets01_101/kinetics/070618/train_avi-480p/riding_a_bike/"
    f = ["4w5sIgC-v4A_000044_000054.avi", "E-JT00ntkUs_000002_000012.avi", "wQpxAGdYuYc_000002_000012.avi"]
    import os
    c = [os.path.join(p, ff) for ff in f]
    video_clips = VideoClips(c)
    from IPython import embed; embed()
