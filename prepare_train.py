import sys
import os
import json
import argparse
import pathlib
from random import shuffle

def prepare_train(exp_dir1, sr2, if_f0_3, version19, spk_id5):
    now_dir = os.getcwd()
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        try:
            names = (
                set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
                & set([name.split(".")[0] for name in os.listdir(feature_dir)])
                & set([name.split(".")[0] for name in os.listdir(f0_dir)])
                & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
            )
        except FileNotFoundError as e:
            print(f"Error reading directories. Ensure extraction has run: {e}")
            names = []
    else:
        try:
            names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
                [name.split(".")[0] for name in os.listdir(feature_dir)]
            )
        except FileNotFoundError as e:
            print(f"Error reading directories. Ensure extraction has run: {e}")
            names = []
            
    if not names:
        print("Warning: No matching files found to train on. Make sure feature extraction finished.")
            
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
            
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    print("Write filelist done")

    # Generate config.json
    if version19 == "v1" or sr2 == "40k":
        config_path = "configs/v1/%s.json" % sr2
    else:
        config_path = "configs/v2/%s.json" % sr2
        
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_path, "r") as f:
            config_data = json.load(f)
            
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                config_data,
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")
        print("Config generated at", config_save_path)
    else:
        print("Config already exists at", config_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_dir", type=str, required=True, help="Experiment name")
    parser.add_argument("-sr", "--sample_rate", type=str, required=True, help="Sample rate (e.g. 40k)")
    parser.add_argument("-f0", "--if_f0", type=int, default=1, help="If f0 (1 or 0)")
    parser.add_argument("-v", "--version", type=str, default="v2", help="Version (v1 or v2)")
    parser.add_argument("-spk", "--spk_id", type=int, default=0, help="Speaker ID")
    args = parser.parse_args()
    
    prepare_train(args.experiment_dir, args.sample_rate, bool(args.if_f0), args.version, args.spk_id)
