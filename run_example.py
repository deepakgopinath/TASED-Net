import sys
import os
import numpy as np
import cv2
import argparse
import torch
import pickle
from model import TASED_v2
from scipy.ndimage.filters import gaussian_filter


def main(args):
    """ read frames in path_indata and generate frame-wise saliency maps in path_output """
    # optional two command-line arguments

    path_indata = args.path_indata
    path_output = args.path_output
    ds_type = args.ds_type
    len_temporal = args.len_temporal
    session_name = args.session_name
    file_weight = args.file_weight
    path_to_test_set_pkl = args.path_to_test_set_pkl
    max_count = args.max_count

    if not os.path.isdir(path_output):
        os.makedirs(path_output)

    model = TASED_v2()

    # load the weight file and copy the parameters
    if os.path.isfile(file_weight):
        print("loading weight file")
        weight_dict = torch.load(file_weight)
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if "module" in name:
                name = ".".join(name.split(".")[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    print(" size? " + name, param.size(), model_dict[name].size())
            else:
                print(" name? " + name)

        print(" loaded")
    else:
        print("weight file?")

    model = model.cuda()
    torch.backends.cudnn.benchmark = False
    model.eval()

    # iterate over the path_indata directory
    list_indata = [d for d in os.listdir(path_indata) if os.path.isdir(os.path.join(path_indata, d))]
    list_indata.sort()
    if ds_type == "nback":
        with open(path_to_test_set_pkl, "rb") as fp:
            test_set_vids = pickle.load(fp)

    import IPython

    IPython.embed(banner1="check")

    ctr = 0
    for dname in list_indata:
        # for each directory in the input folder
        if ds_type == "nback":
            if dname not in test_set_vids:
                continue
        if ctr > max_count:
            break
        print("processing " + dname)
        # all frames listed in the directory
        list_frames = [
            f
            for f in os.listdir(os.path.join(path_indata, dname))
            if os.path.isfile(os.path.join(path_indata, dname, f))
        ]
        list_frames.sort()

        # process in a sliding window fashion
        if len(list_frames) >= 2 * len_temporal - 1:
            path_outdata = os.path.join(path_output, dname)
            if not os.path.isdir(path_outdata):
                os.makedirs(path_outdata)

            snippet = []
            for i in range(len(list_frames)):
                img = cv2.imread(os.path.join(path_indata, dname, list_frames[i]))
                img = cv2.resize(img, (384, 224))
                img = img[..., ::-1]
                snippet.append(img)

                if i >= len_temporal - 1:
                    clip = transform(snippet)

                    process(model, clip, path_outdata, i)

                    # process first (len_temporal-1) frames
                    if i < 2 * len_temporal - 2:
                        process(model, torch.flip(clip, [2]), path_outdata, i - len_temporal + 1)

                    del snippet[0]

        else:
            print(" more frames are needed")

        ctr += 1

    print("Done inference for {} videos".format(max_count))


def transform(snippet):
    """ stack & noralization """
    snippet = np.concatenate(snippet, axis=-1)
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
    snippet = snippet.mul_(2.0).sub_(255).div(255)

    return snippet.view(1, -1, 3, snippet.size(1), snippet.size(2)).permute(0, 2, 1, 3, 4)


def process(model, clip, path_outdata, idx):
    """ process one clip and save the predicted saliency map """
    with torch.no_grad():
        smap = model(clip.cuda()).cpu().data[0]

    smap = (smap.numpy() * 255.0).astype(np.int) / 255.0
    smap = gaussian_filter(smap, sigma=2)
    cv2.imwrite(os.path.join(path_outdata, "%04d.png" % (idx + 1)), (smap / np.max(smap) * 255.0).astype(np.uint8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_indata", default="./DHF1K", help="Directory in which the raw_data is stored",
    )
    parser.add_argument(
        "--path_output", default="./output", help="Directory in which models will be saved",
    )
    parser.add_argument(
        "--path_to_test_set_pkl",
        default="./output/test_split.pkl",
        help="Path to pkl file containing the test set will be stored ",
    )
    parser.add_argument(
        "--file_weight", default="./TASED_updated.pt", help="Path to model file ",
    )
    parser.add_argument(
        "--ds_type", default="DHF1k", help="dataset type used. Currently supporting [DHF1k, nback]",
    )
    parser.add_argument(
        "--len_temporal", type=int, default=32, help="Length of slice used for TASED net",
    )
    parser.add_argument(
        "--max_count", type=int, default=10, help="Length of slice used for TASED net",
    )

    parser.add_argument(
        "--session_name", default="dhf1k_test", help="Wandb session name",
    )

    args = parser.parse_args()
    main(args)

