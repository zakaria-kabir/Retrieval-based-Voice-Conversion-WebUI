#!/usr/bin/env python3
import argparse
import yaml
import sys
import os
import subprocess
import shutil
import tarfile
import json
import re
import csv
import time
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def print_step(message):
    print(f"\n{'='*60}")
    print(f"🚀 {message}")
    print(f"{'='*60}\n")

def run_command(cmd, cwd=None, capture_stdout=None):
    print(f"Executing: {cmd}")
    
    if capture_stdout:
        # Popen to capture stdout to file and print to terminal
        with open(capture_stdout, 'a') as f_out:
            process = subprocess.Popen(
                cmd, shell=True, cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True
            )
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                f_out.write(line)
                f_out.flush()
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
    else:
        subprocess.run(cmd, shell=True, check=True, cwd=cwd)

class RVCPipeline:
    def __init__(self, config):
        self.config = config
        self.repo_root = config['rvc_repo_root']
        self.backup_root = config['backup_root']
        self.model_name = config['model_name']
        self.dataset_tar = config.get('dataset_tar')
        self.dataset_dir = config.get('dataset_extract_dir')
        
    def preflight_check(self):
        print_step("Preflight Checks")
        
        # 1. Check repo root
        if not os.path.exists(self.repo_root) or not os.path.exists(os.path.join(self.repo_root, "infer")):
            print(f"❌ RVC Repo not found or invalid at: {self.repo_root}")
            sys.exit(1)
        print(f"✅ RVC Repo found at {self.repo_root}")

        # 2. Check Pretrained Models
        v2_path = os.path.join(self.repo_root, f"assets/pretrained_v2")
        sr = self.config['sample_rate']
        required_models = [
            os.path.join(v2_path, f"f0G{sr}.pth"),
            os.path.join(v2_path, f"f0D{sr}.pth"),
            os.path.join(self.repo_root, "assets/hubert/hubert_base.pt"),
            os.path.join(self.repo_root, "assets/rmvpe/rmvpe.pt")
        ]
        
        missing = False
        for m in required_models:
            if not os.path.exists(m):
                print(f"❌ Missing pretrained model: {m}")
                missing = True
            else:
                print(f"✅ Found model: {os.path.basename(m)}")
        if missing:
            sys.exit(1)
            
        print("✅ Preflight checks passed.")

    def extract_dataset(self):
        print_step("Extracting Dataset")
        if not self.dataset_tar or not os.path.exists(self.dataset_tar):
            print(f"❌ Dataset tarball not found: {self.dataset_tar}")
            sys.exit(1)
            
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        # Cleanup existing extract dir if needed
        model_dataset_dir = os.path.join(self.dataset_dir, self.model_name)
        if os.path.exists(model_dataset_dir):
            print(f"Cleaning up existing directory: {model_dataset_dir}")
            shutil.rmtree(model_dataset_dir)
            
        print(f"Extracting {self.dataset_tar} to {self.dataset_dir}")
        run_command(f"tar -xvf {self.dataset_tar} -C {self.dataset_dir}")
        
        if not os.path.exists(model_dataset_dir):
            print(f"⚠️ Warning: Expected extracted folder {model_dataset_dir} not found. Ensure tarball contains a folder named {self.model_name}")

    def cleanup_dataset(self):
        print_step("Cleaning up extracted dataset")
        model_dataset_dir = os.path.join(self.dataset_dir, self.model_name)
        if os.path.exists(model_dataset_dir):
            shutil.rmtree(model_dataset_dir)
            print(f"✅ Deleted {model_dataset_dir}")
        else:
            print(f"No dataset directory found at {model_dataset_dir} to clean up.")

    def train_pipeline(self):
        self.preflight_check()
        self.extract_dataset()
        
        print_step("Starting Training Pipeline")
        # Explicit thread management to prevent CPU thrashing when running multiple instances
        default_threads = max(1, os.cpu_count() // len(self.config.get('gpu', '0').split('-')))
        threads = self.config.get('threads', default_threads)
        
        # Limit OpenMP threads so FAISS index training doesn't eat all 80 cores per instance
        os.environ["OMP_NUM_THREADS"] = str(threads)
        
        sr_num = int(self.config['sample_rate'].replace('k', '')) * 1000
        logs_dir = os.path.join(self.repo_root, f"logs/{self.model_name}")
        os.makedirs(logs_dir, exist_ok=True)
        dataset_path = os.path.join(self.dataset_dir, self.model_name)
        
        # 1. Preprocess
        print("\n--- 1. Preprocess Dataset ---")
        run_command(
            f"python infer/modules/train/preprocess.py {dataset_path} {sr_num} {threads} {logs_dir} {self.config['no_parallel']} 3.7",
            cwd=self.repo_root
        )
        
        # 2. Extract f0
        print("\n--- 2. Extract F0 ---")
        f0_method = self.config['f0_method']
        if f0_method == "rmvpe_gpu":
            gpus_rmvpe = self.config['gpu'].split("-")
            leng = len(gpus_rmvpe)
            for idx, n_g in enumerate(gpus_rmvpe):
                run_command(
                    f"python infer/modules/train/extract/extract_f0_rmvpe.py {leng} {idx} {n_g} {logs_dir} True",
                    cwd=self.repo_root
                )
        else:
            run_command(
                f"python infer/modules/train/extract/extract_f0_print.py {logs_dir} {threads} {f0_method}",
                cwd=self.repo_root
            )
            
        # 3. Extract Features
        print("\n--- 3. Extract Features ---")
        gpus = self.config['gpu'].split('-')
        leng = len(gpus)
        for idx, n_g in enumerate(gpus):
            run_command(
                f"python infer/modules/train/extract_feature_print.py cuda {leng} {idx} {n_g} {logs_dir} {self.config['version']} True",
                cwd=self.repo_root
            )
            
        # 4. Prepare Train
        print("\n--- 4. Prepare Train (filelist + config) ---")
        run_command(
            f"python prepare_train.py -e {self.model_name} -sr {self.config['sample_rate']} -f0 {1 if self.config['use_f0'] else 0} -v {self.config['version']} -spk {self.config['speaker_id']}",
            cwd=self.repo_root
        )
        
        # 5. Train Model
        print("\n--- 5. Train Model ---")
        save_latest = 1 if self.config['save_latest_only'] else 0
        cache_gpu = 1 if self.config['cache_data_in_gpu'] else 0
        save_weights = 1 if self.config['save_every_weights'] else 0
        use_f0 = 1 if self.config['use_f0'] else 0
        
        train_cmd = (
            f"python infer/modules/train/train.py "
            f"-e {self.model_name} "
            f"-sr {self.config['sample_rate']} "
            f"-f0 {use_f0} "
            f"-bs {self.config['batch_size']} "
            f"-g {self.config['gpu']} "
            f"-te {self.config['total_epoch']} "
            f"-se {self.config['save_every']} "
            f"-pg assets/pretrained_v2/f0G{self.config['sample_rate']}.pth "
            f"-pd assets/pretrained_v2/f0D{self.config['sample_rate']}.pth "
            f"-l {save_latest} -c {cache_gpu} -sw {save_weights} -v {self.config['version']}"
        )
        
        train_log_file = os.path.join(logs_dir, "train_stdout.log")
        run_command(train_cmd, cwd=self.repo_root, capture_stdout=train_log_file)
        
        # 6. Build FAISS Index
        print("\n--- 6. Build FAISS Index ---")
        run_command(
            f"python train_index.py -e {self.model_name} -v {self.config['version']}",
            cwd=self.repo_root
        )
        
        # 7. Analyze Log
        self.analyze_log(train_log_file)
        
        # 8. Backup
        self.backup()
        
        # 9. Cleanup
        self.cleanup_dataset()
        print_step("🎉 Full Training Pipeline Completed!")

    def _minmax_norm(self, series):
        """Min-max normalize a pandas series"""
        return (series - series.min()) / (series.max() - series.min() + 1e-8)

    def analyze_log(self, log_path=None):
        print_step("Analyzing Training Log")
        
        if not log_path:
            log_path = os.path.join(self.repo_root, f"logs/{self.model_name}/train.log")
            if not os.path.exists(log_path):
                 log_path = os.path.join(self.repo_root, f"logs/{self.model_name}/train_stdout.log")
                 
        if not os.path.exists(log_path):
            print(f"❌ Log file not found at {log_path}")
            sys.exit(1)
            
        print(f"Parsing log file: {log_path}")
        
        rows = []
        pattern = re.compile(
            r".*loss_disc=([\d\.]+),\s*loss_gen=([\d\.]+),\s*loss_fm=([\d\.]+),\s*loss_mel=([\d\.]+),\s*loss_kl=([\d\.]+)"
        )
        
        with open(log_path, "r", encoding="utf-8") as f:
            for ln in f:
                m = pattern.search(ln)
                if m:
                    rows.append({
                        "step": len(rows),
                        "loss_disc": float(m.group(1)),
                        "loss_gen": float(m.group(2)),
                        "loss_fm": float(m.group(3)),
                        "loss_mel": float(m.group(4)),
                        "loss_kl": float(m.group(5)),
                    })
                    
        if not rows:
            print("❌ No loss lines found in the log.")
            return

        df = pd.DataFrame(rows)
        
        backup_dir = os.path.join(self.backup_root, self.model_name)
        os.makedirs(backup_dir, exist_ok=True)
        
        csv_path = os.path.join(backup_dir, "training_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved metrics to {csv_path}")
        
        # Calculate composite score (lowest is best)
        # We normalize all losses so they contribute equally
        norm_mel = self._minmax_norm(df['loss_mel'])
        norm_fm = self._minmax_norm(df['loss_fm'])
        norm_gen = self._minmax_norm(df['loss_gen'])
        norm_kl = self._minmax_norm(df['loss_kl'])
        # For disc, ideally it's balanced (~3.0-4.0), not too high or too low, but for simplicity we treat lower as better here, or just ignore disc in composite score since generator metrics matter more for audio quality.
        # Let's use mel + fm + gen + kl
        composite_score = norm_mel + norm_fm + norm_gen + norm_kl
        df['composite_score'] = composite_score
        
        # Get top 3
        best_rows = df.nsmallest(3, 'composite_score')
        best_candidates = best_rows.to_dict(orient='records')
        
        json_path = os.path.join(backup_dir, "best_checkpoints.json")
        with open(json_path, 'w') as f:
            json.dump(best_candidates, f, indent=4)
        print(f"✅ Saved top-3 best checkpoint candidates to {json_path}")
        
        print("\n🏆 Top 3 Best Candidates (by composite score):")
        for i, c in enumerate(best_candidates):
            print(f"  #{i+1} : step {c['step']} | composite={c['composite_score']:.4f} | mel={c['loss_mel']:.4f} | fm={c['loss_fm']:.4f}")
            
        # Plotting
        try:
            fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
            
            axs[0].plot(df['step'], df['loss_mel'], label='loss_mel', color='blue')
            axs[0].plot(df['step'], df['loss_fm'], label='loss_fm', color='cyan')
            axs[0].set_title('Spectral Quality')
            axs[0].legend()
            axs[0].grid(True)
            
            axs[1].plot(df['step'], df['loss_gen'], label='loss_gen', color='green')
            axs[1].plot(df['step'], df['loss_disc'], label='loss_disc', color='red')
            axs[1].set_title('Adversarial Balance')
            axs[1].legend()
            axs[1].grid(True)
            
            axs[2].plot(df['step'], df['loss_kl'], label='loss_kl', color='purple')
            axs[2].set_title('Latent Divergence')
            axs[2].legend()
            axs[2].grid(True)
            
            plt.xlabel('Step')
            plt.tight_layout()
            
            plot_path = os.path.join(backup_dir, "training_metrics.png")
            plt.savefig(plot_path)
            print(f"✅ Saved multi-panel plot to {plot_path}")
        except Exception as e:
            print(f"⚠️ Failed to generate plot: {e}")

    def backup(self):
        print_step("Backing up Model Files")
        backup_dir = os.path.join(self.backup_root, self.model_name)
        os.makedirs(backup_dir, exist_ok=True)
        
        import glob
        
        # 1. Inference Weights
        weight_files = glob.glob(os.path.join(self.repo_root, f"assets/weights/{self.model_name}*.pth"))
        for f in weight_files:
            try:
                shutil.copy2(f, backup_dir)
            except Exception as e:
                print(f"⚠️ Failed to copy weight {f}: {e}")
        print(f"✅ Copied {len(weight_files)} inference weights.")
        
        # 2. Latest Checkpoints for Resume
        log_dir = os.path.join(self.repo_root, f"logs/{self.model_name}")
        if os.path.exists(log_dir):
            g_files = glob.glob(f"{log_dir}/G_*.pth")
            d_files = glob.glob(f"{log_dir}/D_*.pth")
            if g_files and d_files:
                latest_g = sorted(g_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
                latest_d = sorted(d_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
                try:
                    shutil.copy2(latest_g, backup_dir)
                    shutil.copy2(latest_d, backup_dir)
                    print(f"✅ Copied latest resume checkpoints: {os.path.basename(latest_g)}, {os.path.basename(latest_d)}")
                except Exception as e:
                     print(f"⚠️ Failed to copy latest checkpoints: {e}")
            
            # 3. FAISS Index
            index_files = glob.glob(f"{log_dir}/added_IVF*.index")
            if index_files:
                try:
                    shutil.copy2(index_files[0], backup_dir)
                    print(f"✅ Copied FAISS index: {os.path.basename(index_files[0])}")
                except Exception as e:
                    print(f"⚠️ Failed to copy index: {e}")
                    
            # 4. Config & filelist
            for f_name in ["config.json", "filelist.txt", "train_stdout.log", "train.log"]:
                f_path = os.path.join(log_dir, f_name)
                if os.path.exists(f_path):
                    try:
                        shutil.copy2(f_path, backup_dir)
                    except Exception as e:
                        print(f"⚠️ Failed to copy {f_name}: {e}")
        
        print(f"🎉 Backup complete to: {backup_dir}")

    def restore(self):
        print_step("Restoring Model Files for Inference")
        backup_dir = os.path.join(self.backup_root, self.model_name)
        if not os.path.exists(backup_dir):
             print(f"❌ Backup folder not found: {backup_dir}")
             sys.exit(1)
             
        import glob
        # Restore latest weight (if multiple pick highest epoch or just base model name)
        weights = glob.glob(f"{backup_dir}/{self.model_name}*.pth")
        if weights:
            # simple sort to pick the 'best' or base
            target_w = os.path.join(self.repo_root, "assets/weights")
            os.makedirs(target_w, exist_ok=True)
            for w in weights:
                if "G_" not in os.path.basename(w) and "D_" not in os.path.basename(w):
                    shutil.copy2(w, target_w)
                    print(f"✅ Restored weight: {os.path.basename(w)} to assets/weights/")
        else:
             print("⚠️ No inference weights found in backup.")
             
        # Restore index
        indices = glob.glob(f"{backup_dir}/added_IVF*.index") + glob.glob(f"{backup_dir}/{self.model_name}_IVF*.index")
        if indices:
            target_i = os.path.join(self.repo_root, f"logs/{self.model_name}")
            os.makedirs(target_i, exist_ok=True)
            shutil.copy2(indices[0], target_i)
            print(f"✅ Restored index: {os.path.basename(indices[0])} to logs/{self.model_name}/")
        else:
             print("⚠️ No FAISS index found in backup.")

    def inference(self):
        print_step("Inference Pipeline")
        inf_config = self.config['inference']
        
        # Resolve model input
        input_audio_path = inf_config['input_audio']
        if not input_audio_path:
             print("❌ No input_audio provided in config.")
             sys.exit(1)
             
        is_batch = os.path.isdir(input_audio_path)
        if is_batch:
            audio_files = [os.path.join(input_audio_path, f) for f in os.listdir(input_audio_path) if f.lower().endswith('.wav')]
        else:
            audio_files = [input_audio_path]
            
        if not audio_files:
            print(f"❌ No WAV files found in: {input_audio_path}")
            sys.exit(1)
            
        # Resolve paths
        from argparse import Namespace
        from infer.modules.vc.modules import VC
        from configs.config import Config as RVCConfig
        import glob
        
        # Temporarily change dir to repo root so things load cleanly
        original_cwd = os.getcwd()
        os.chdir(self.repo_root)
        sys.path.append(self.repo_root)

        # Build RVC Config
        rvc_cfg = RVCConfig()
        rvc_cfg.device = f"cuda:{self.config['gpu'].split('-')[0]}"
        vc = VC(rvc_cfg)
        
        # Load weights
        weight_name = inf_config['model_weight']
        if not weight_name:
             backup_dir = os.path.join(self.backup_root, self.model_name)
             weights = glob.glob(f"{backup_dir}/{self.model_name}*.pth")
             valid_w = [w for w in weights if "G_" not in os.path.basename(w) and "D_" not in os.path.basename(w)]
             if valid_w:
                 weight_name = os.path.basename(valid_w[0])
                 # Ensure it's inside assets/weights
                 tgt = os.path.join("assets/weights", weight_name)
                 if not os.path.exists(tgt):
                     shutil.copy2(valid_w[0], tgt)
             else:
                 weight_name = f"{self.model_name}.pth" # fallback
                 
        print(f"Loading Model: {weight_name}")
        sid_info = vc.get_vc(weight_name)
        
        index_name = inf_config['index_file']
        if not index_name:
             backup_dir = os.path.join(self.backup_root, self.model_name)
             indices = glob.glob(f"{backup_dir}/*IVF*.index")
             if indices:
                 index_name = indices[0]
                 tgt = os.path.join(f"logs/{self.model_name}", os.path.basename(index_name))
                 os.makedirs(f"logs/{self.model_name}", exist_ok=True)
                 if not os.path.exists(tgt):
                     shutil.copy2(indices[0], tgt)
                 index_name = tgt
        
        print(f"Loading Index: {index_name}")

        out_dir = inf_config['output_dir']
        os.makedirs(out_dir, exist_ok=True)
        
        batch_log_rows = []
        randomize = inf_config.get('randomize_params', False)
        rng = inf_config.get('random_ranges', {})
        
        for f_in in audio_files:
            print(f"\nProcessing: {f_in}")
            
            # Setup params
            params = {
                'f0_up_key': inf_config['f0_up_key'],
                'f0_method': inf_config['f0_method'],
                'index_rate': inf_config['index_rate'],
                'filter_radius': inf_config['filter_radius'],
                'resample_sr': inf_config['resample_sr'],
                'rms_mix_rate': inf_config['rms_mix_rate'],
                'protect': inf_config['protect']
            }
            
            if randomize:
                params['f0_up_key'] = random.randint(rng.get('f0_up_key', [0,0])[0], rng.get('f0_up_key', [0,0])[1])
                params['index_rate'] = random.uniform(rng.get('index_rate', [0,0])[0], rng.get('index_rate', [0,0])[1])
                params['filter_radius'] = random.randint(rng.get('filter_radius', [0,0])[0], rng.get('filter_radius', [0,0])[1])
                params['rms_mix_rate'] = random.uniform(rng.get('rms_mix_rate', [0,0])[0], rng.get('rms_mix_rate', [0,0])[1])
                params['protect'] = random.uniform(rng.get('protect', [0,0])[0], rng.get('protect', [0,0])[1])
                print(f"  Randomized Params: {params}")
                
            info = vc.vc_single(
                1, # sid (not used directly if model loaded)
                f_in,
                params['f0_up_key'],
                inf_config.get('f0_file', None),
                params['f0_method'],
                index_name,
                "", # index2
                params['index_rate'],
                params['filter_radius'],
                params['resample_sr'],
                params['rms_mix_rate'],
                params['protect']
            )
            
            # Save Output
            fmt = inf_config['output_format']
            out_file = os.path.join(out_dir, f"{os.path.basename(f_in).rsplit('.',1)[0]}_rvc_{self.model_name}.{fmt}")
            
            msg, audio_opt = info
            if "Success" in msg and audio_opt[0] is not None:
                import soundfile as sf
                from io import BytesIO
                from infer.lib.audio import wav2
                
                tgt_sr, data = audio_opt
                if fmt in ["wav", "flac"]:
                    sf.write(out_file, data, tgt_sr)
                else:
                    with BytesIO() as wavf:
                         sf.write(wavf, data, tgt_sr, format="wav")
                         wavf.seek(0, 0)
                         with open(out_file, "wb") as outf:
                              wav2(wavf, outf, fmt)
                              
                print(f"✅ Saved to: {out_file}")
                
                # Log batch stats
                if is_batch:
                   batch_log_rows.append({
                       'file': os.path.basename(f_in),
                       'out_file': os.path.basename(out_file),
                       **params
                   })
            else:
                 print(f"❌ Failed to process: {msg}")

        if is_batch and batch_log_rows:
            csv_path = os.path.join(out_dir, "batch_inference_log.csv")
            pd.DataFrame(batch_log_rows).to_csv(csv_path, index=False)
            print(f"✅ Batch inference log saved to: {csv_path}")
            
        os.chdir(original_cwd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RVC Pipeline CLI")
    parser.add_argument("command", choices=["train", "infer", "backup", "restore", "analyze"], help="Command to run")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    parser.add_argument("--dry-run", action="store_true", help="Only run preflight checks (for train command)")
    parser.add_argument("--log-file", help="Path to specific log file to analyze (override config)")
    
    args = parser.parse_args()
    cfg = load_config(args.config)
    
    pipeline = RVCPipeline(cfg)
    
    if args.command == "train":
        if args.dry_run:
            pipeline.preflight_check()
        else:
            pipeline.train_pipeline()
    elif args.command == "infer":
        pipeline.inference()
    elif args.command == "backup":
        pipeline.backup()
    elif args.command == "restore":
        pipeline.restore()
    elif args.command == "analyze":
        pipeline.analyze_log(args.log_file)
