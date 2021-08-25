import os
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def transform(snippet):
    """ stack & noralization """
    snippet = np.concatenate(snippet, axis=-1)  # H, W, 3*len_temporal
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()  # 3*len_temporal, H, W
    snippet = snippet.mul_(2.0).sub_(255).div(255)  # normalize to -1.0, 1.0 3*len_temporal, H, W
    snippet = snippet.view(-1, 3, snippet.size(1), snippet.size(2)).permute(1, 0, 2, 3)  # 3, len_temporal, H, W
    return snippet


class NBackDataset(Dataset):
    def __init__(self, path_data, len_snippet, data_list):

        """
        data_list: list of strs. all videos that will be part of the dataset
        """
        self.path_data = path_data
        self.len_snippet = len_snippet
        with open("nback_list_num_frames_all.pkl", "rb") as fp:
            self.nback_list_num_frames_all_dict = pickle.load(fp)

        self.video_data_list = data_list

    def __len__(self):
        return len(self.video_data_list)

    def __getitem__(self, idx):
        file_name = self.video_data_list[idx]
        path_clip = os.path.join(self.path_data, "stimuli_chunked_10", file_name)
        # path to the smoothed fixation maps for each frame.
        path_annt = os.path.join(self.path_data, "fixation_maps", file_name)

        # -1 because the last frame sometimes is empty
        start_idx = np.random.randint(0, (self.nback_list_num_frames_all_dict[file_name] - 1) - self.len_snippet + 1)
        v = np.random.random()
        clip = []  # list of input frames

        for i in range(self.len_snippet):
            # video frame
            img = cv2.imread(os.path.join(path_clip, "%04d.png" % (start_idx + i + 1)))
            # resized video frame. can potentially cache this on the fly to reduce time spent on resizing
            img = cv2.resize(img, (384, 224))  # fixed size for TASED net input.
            # BGR to RGB?
            img = img[..., ::-1]
            if v < 0.5:
                # horizontal flip?
                img = img[:, ::-1, ...]
            clip.append(img)

        # get fixation map for T. TASED-Net only predicts fixation for T
        annt = cv2.imread(os.path.join(path_annt, "%04d.png" % (start_idx + self.len_snippet)))  # (H, W, 3)
        annt = cv2.cvtColor(annt, cv2.COLOR_BGR2GRAY)  # gray scale (H, W)
        # cache reszed fixation map as well to reduce time.
        annt = cv2.resize(annt, (384, 224))  # H, W still in 0,255 range. normalization happens in loss functions
        if v < 0.5:
            # horizontal flip fixation map as well
            annt = annt[:, ::-1]

        # transform(clip) is C, T, H, W,
        # torch.from_numpy(annt.copy()).contiguous().float() is H, W
        return transform(clip), torch.from_numpy(annt.copy()).contiguous().float()


class DHF1KDataset(Dataset):
    def __init__(self, path_data, len_snippet):
        self.path_data = path_data
        self.len_snippet = len_snippet
        # number of frames in each training video. First 600 videos are training videos.
        self.list_num_frame = [int(row[0]) for row in csv.reader(open("DHF1K_num_frame_train.csv", "r"))]

    def __len__(self):
        # length of dataset is equal to the number of training videos used.
        return len(self.list_num_frame)

    def __getitem__(self, idx):
        # should return video frame sequence and the associated fixation map for the to be predicted frame
        file_name = "%04d" % (idx + 1)
        # path to folder which contains the pngs extracted from video
        path_clip = os.path.join(self.path_data, "video", file_name)
        # path to the smoothed fixation maps for each frame.
        path_annt = os.path.join(self.path_data, "annotation", file_name, "maps")

        start_idx = np.random.randint(0, self.list_num_frame[idx] - self.len_snippet + 1)

        v = np.random.random()
        clip = []  # list of input frames
        for i in range(self.len_snippet):
            # video frame
            img = cv2.imread(os.path.join(path_clip, "%04d.png" % (start_idx + i + 1)))
            # resized video frame. can potentially cache this on the fly to reduce time spent on resizing
            img = cv2.resize(img, (384, 224))  # fixed size for TASED net input.

            # BGR to RGB?
            img = img[..., ::-1]
            if v < 0.5:
                # horizontal flip
                img = img[:, ::-1, ...]
            clip.append(img)

        # get fixation map for T. TASED-Net only predicts fixation for T
        annt = cv2.imread(os.path.join(path_annt, "%04d.png" % (start_idx + self.len_snippet)), 0)
        # cache reszed fixation map as well to reduce time.
        annt = cv2.resize(annt, (384, 224))  # still in 0,255 range. normalization happens in loss functions
        if v < 0.5:
            # horizontal flip fixation map as well
            annt = annt[:, ::-1]

        return transform(clip), torch.from_numpy(annt.copy()).contiguous().float()


# Reference: gist.github.com/MFreidank/821cc87b012c53fade03b0c7aba13958
class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


# ** Update **
# Please consider using the following sampler during training instead of the above InfiniteDataLoader.
# You can simply refer to: https://github.com/MichiganCOG/Gaze-Attention
# Reference: https://github.com/facebookresearch/detectron2
class trainingSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size):
        self.size = size

    def _infinite_indices(self):
        g = torch.Generator()
        while True:
            yield from torch.randperm(self.size, generator=g)

    def __iter__(self):
        yield from itertools.islice(self._infinite_indices(), 0, None, 1)
