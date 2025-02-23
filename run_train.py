import sys
import os
import numpy as np
import cv2
import time
import random
import argparse
import pickle
from datetime import timedelta
import torch
import wandb
from model import TASED_v2
from loss import KLDLoss
from dataset import DHF1KDataset, InfiniteDataLoader, NBackDataset
from torch.utils.data import DataLoader
from itertools import islice


def main(args):
    """ concise script for training """
    # optional two command-line arguments
    wandb.login()

    path_indata = args.path_indata
    path_output = args.path_output
    ds_type = args.ds_type
    len_temporal = args.len_temporal
    testing_frequency = args.testing_frequency
    session_name = args.session_name

    print("Path to data folder ", path_indata)
    num_gpu = 2
    pile = 5
    batch_size = 8
    num_iters = 1000
    wandb.init(project="TASED_net", id=session_name, config={"dataset": ds_type, "batch_size": batch_size})

    file_weight = "./S3D_kinetics400.pt"
    path_output = os.path.join(path_output, time.strftime("%m-%d_%H-%M-%S"))
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
            if "base." in name:
                bn = int(name.split(".")[1])
                sn_list = [0, 5, 8, 14]
                sn = sn_list[0]
                if bn >= sn_list[1] and bn < sn_list[2]:
                    sn = sn_list[1]
                elif bn >= sn_list[2] and bn < sn_list[3]:
                    sn = sn_list[2]
                elif bn >= sn_list[3]:
                    sn = sn_list[3]
                name = ".".join(name.split(".")[2:])
                name = "base%d.%d." % (sn_list.index(sn) + 1, bn - sn) + name
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

    # parameter setting for fine-tuning
    params = []
    for key, value in dict(model.named_parameters()).items():
        if "convtsp" in key:
            params += [{"params": [value], "key": key + "(new)"}]
        else:
            params += [{"params": [value], "lr": 0.001, "key": key}]

    optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=2e-7)
    criterion = KLDLoss()

    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    torch.backends.cudnn.benchmark = False
    model.train()
    if ds_type == "DHF1k":
        dhf1k_ds = DHF1KDataset(path_indata, len_temporal)
        train_loader = InfiniteDataLoader(dhf1k_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    elif ds_type == "nback":
        with open("nback_list_num_frames_all.pkl", "rb") as fp:
            nback_list_num_frames_all_dict = pickle.load(fp)

        # remove video whose length is less than 32
        nback_list_num_frames_all_dict = {k: v for k, v in nback_list_num_frames_all_dict.items() if v >= len_temporal}
        num_all_videos = len(nback_list_num_frames_all_dict.keys())
        print("Number of long enough videos ", num_all_videos)
        all_video_list = list(nback_list_num_frames_all_dict.keys())
        train_video_list = random.sample(list(nback_list_num_frames_all_dict.keys()), int(0.9 * num_all_videos))
        test_video_list = list(set(all_video_list) - set(train_video_list))

        # save test split
        with open(os.path.join(path_output, "test_split.pkl"), "wb") as fp:
            pickle.dump(test_video_list, fp)

        print("Length before deleting sanity challenge ", len(train_video_list))

        if "sanity_check_challenge_1-of-3" in train_video_list:
            train_video_list.remove("sanity_check_challenge_1-of-3")
        if "sanity_check_challenge_2-of-3" in train_video_list:
            train_video_list.remove("sanity_check_challenge_2-of-3")
        if "sanity_check_challenge_3-of-3" in train_video_list:
            train_video_list.remove("sanity_check_challenge_3-of-3")

        if "sanity_check_challenge_1-of-3" in test_video_list:
            test_video_list.remove("sanity_check_challenge_1-of-3")
        if "sanity_check_challenge_2-of-3" in test_video_list:
            test_video_list.remove("sanity_check_challenge_2-of-3")
        if "sanity_check_challenge_3-of-3" in test_video_list:
            test_video_list.remove("sanity_check_challenge_3-of-3")

        print("Length  deleting sanity challenge ", len(train_video_list))

        nback_train_ds = NBackDataset(path_indata, len_temporal, train_video_list)
        nback_test_ds = NBackDataset(path_indata, len_temporal, test_video_list)

        train_loader = InfiniteDataLoader(nback_train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
        test_loader = DataLoader(nback_test_ds, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)

    i, step, test_step = 0, 0, 0
    loss_sum = 0
    loss_sum_test = 0.0
    start_time = time.time()
    # no testing loop during training.
    for clip, annt in islice(train_loader, num_iters * pile):
        with torch.set_grad_enabled(True):
            output = model(clip.cuda())
            loss = criterion(output, annt.cuda())

        loss_sum += loss.detach().item()
        loss.backward()
        # aggregation steps before backprp
        if (i + 1) % pile == 0:
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            # whole process takes less than 3 hours
            print(
                "training iteration: [%4d/%4d], loss: %.4f, %s"
                % (step, num_iters, loss_sum / pile, timedelta(seconds=int(time.time() - start_time))),
                flush=True,
            )
            wandb.log({"training_loss": loss_sum / pile, "iteration_time": int(time.time() - start_time)})
            loss_sum = 0

            # adjust learning rate
            if step in [750, 950]:
                for opt in optimizer.param_groups:
                    if "new" in opt["key"]:
                        opt["lr"] *= 0.1

            if step % 25 == 0:
                torch.save(model.state_dict(), os.path.join(path_output, "iter_%04d.pt" % step))

            if ds_type == "nback":
                if step % testing_frequency == 0:
                    # testing phase
                    model.train(False)

                    for clip_test, annt_test in test_loader:
                        with torch.set_grad_enabled(False):
                            output_test = model(clip_test.cuda())
                            loss_test = criterion(output_test, annt_test.cuda())

                        loss_sum_test += loss_test.detach().item()
                        print(
                            "test_iteration_iteration: [%4d/%4d], loss: %.4f, %s"
                            % (test_step, num_iters, loss_sum_test, timedelta(seconds=int(time.time() - start_time))),
                            flush=True,
                        )
                        wandb.log({"test_loss": loss_sum_test})
                        loss_sum_test = 0
                        test_step += 1

                    model.train(True)

        i += 1

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_indata", default="./DHF1K", help="Directory in which the raw_data is stored",
    )
    parser.add_argument(
        "--path_output", default="./output", help="Directory in which models will be saved",
    )
    parser.add_argument(
        "--ds_type", default="DHF1k", help="dataset type used. Currently supporting [DHF1k, nback]",
    )
    parser.add_argument(
        "--len_temporal", type=int, default=32, help="Length of slice used for TASED net",
    )
    parser.add_argument(
        "--testing_frequency", type=int, default=100, help="Frequency of running testing loop",
    )
    parser.add_argument(
        "--test_split", type=float, default=0.2, help="Fraction of ds used for test split.",
    )
    parser.add_argument(
        "--session_name", default="dhf1k_train", help="Wandb session name",
    )

    args = parser.parse_args()
    main(args)
