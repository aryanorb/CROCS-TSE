# -*- coding: utf-8 -*-
"""
Created on Wed Jan 7 10:52:20 2026

@author: swhan
"""

import os
import argparse
import datetime
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi

from dataloader import LibriMixDevTestSet
from models.model import CROCS


# -----------------------------
# Utils
# -----------------------------
def get_timestamp() -> str:
    return datetime.datetime.now().strftime("%y%m%d-%H%M")


def set_seed(seed: int) -> None:
    
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_sisdr(reference: np.ndarray, estimation: np.ndarray, eps: float = 1e-8) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    estimation = np.asarray(estimation, dtype=np.float64)

    reference -= reference.mean()
    estimation -= estimation.mean()

    scale = np.dot(reference, estimation) / (np.dot(reference, reference) + eps)
    projection = scale * reference
    error = estimation - projection

    return float(10.0 * np.log10((np.sum(projection**2) + eps) / (np.sum(error**2) + eps)))


def rms_match(estimated: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms_est = np.sqrt(np.mean(estimated**2) + eps)
    rms_tgt = np.sqrt(np.mean(target**2) + eps)
    return estimated * (rms_tgt / rms_est)


def none_or_float(value: str) -> Optional[float]:
    if value.lower() == "none":
        return None
    try:
        return float(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid float value: '{value}'") from e


def setup_logger(log_dir: str, name: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear() 

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%y-%m-%d %H:%M:%S")

    log_path = os.path.join(log_dir, f"{name}.txt")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)


    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Config
# -----------------------------

@dataclass
class Args:
    seed: int
    train_lengths: float
    reference_lengths: Optional[float]

    fusion_type: str
    temperature: float
    R_coarse: int
    R_fine: int
    M: int
    H: int

    compression_factor: float
    sr: int

    Libri2Mix_path: str
    Libri2Mix_test_list: str

    save_path: str
    converged_model: str

    log_path: str
    recon_path: str

    n_fft: int
    window_length: int
    hop_size: int

    cuda_visible_devices: str
    rms_normal: bool
    inference_task: str
    save_audio: bool


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Target Speaker Extraction with Cross-Correlation for Complex Spectra and Dual Post-Refinements"
    )

    # environment
    parser.add_argument("--cuda_visible_devices", default="0", type=str)
    parser.add_argument("--seed", default=42, type=int)

    # data / task
    parser.add_argument("--train_lengths", default=4.0, type=float)
    parser.add_argument("--reference_lengths", default=None, type=none_or_float)

    parser.add_argument("--Libri2Mix_path", default="/data3/data/LibriMix", type=str)
    parser.add_argument("--Libri2Mix_test_list", default="./speakerbeam_test_mixture2enrollment.txt", type=str)
    parser.add_argument("--inference_task", default="mix_clean", choices=["mix_clean", "mix_both"], type=str)

    # model
    parser.add_argument("--fusion_type", default="PSM", choices=["PSM", "IRM", "CIENet"], type=str)
    parser.add_argument("--temperature", default=4.0, type=float)
    parser.add_argument("--R_coarse", default=3, type=int)
    parser.add_argument("--R_fine", default=1, type=int)
    parser.add_argument("--M", default=4, type=int)
    parser.add_argument("--H", default=80, type=int)

    # signal
    parser.add_argument("--compression_factor", default=0.3, type=float)
    parser.add_argument("--sr", default=8000, type=int)
    parser.add_argument("--n_fft", default=256, type=int)
    parser.add_argument("--window_length", default=160, type=int)
    parser.add_argument("--hop_size", default=80, type=int)

    # io
    parser.add_argument("--save_path", default="./checkpoint", type=str)
    parser.add_argument("--converged_model", default="model.pth", type=str)
    parser.add_argument("--log_path", default="./log", type=str)
    parser.add_argument("--recon_path", default="./recon", type=str)
    parser.add_argument("--rms_normal", action="store_true", help="RMS-match estimated audio to target RMS")
    parser.add_argument("--save_audio", action="store_true", help="Save reconstructed audios")

    return parser


# -----------------------------
# Core
# -----------------------------
def load_model(args: Args, device: torch.device) -> CROCS:
    model_config = dict(
        R_coarse=args.R_coarse,
        R_fine=args.R_fine,
        M=args.M,
        H=args.H,
        compression_factor=args.compression_factor,
        fusion_type=args.fusion_type,
        temperature=args.temperature,
        n_fft=args.n_fft,
        window_length=args.window_length,
        hop_size=args.hop_size,
        device=device,
    )
    model = CROCS(**model_config).to(device)

    ckpt_path = os.path.join(args.save_path, args.converged_model)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "model" not in ckpt:
        raise KeyError(f"Checkpoint does not contain 'model' key: {ckpt_path}")

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


def build_dataset(args: Args) -> LibriMixDevTestSet:
    return LibriMixDevTestSet(
        data_list=args.Libri2Mix_test_list,
        root=args.Libri2Mix_path,
        subset="test",
        task=args.inference_task,
        enrollment_lengths=None,
        sample_rate=args.sr,
    )


@torch.inference_mode()
def run_inference(
    model: CROCS,
    dataset: LibriMixDevTestSet,
    device: torch.device,
    logger: logging.Logger,
    recon_path: str,
    sr: int,
    rms_normal: bool,
    save_audio: bool,
) -> Dict[str, float]:
    safe_makedirs(recon_path)

    pesq_list: List[float] = []
    estoi_list: List[float] = []
    sisdr_list: List[float] = []

    for idx in tqdm(range(len(dataset)), desc="Inference"):
        mixture, target, enrollment = dataset[idx]

        mixture = mixture.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        enrollment = enrollment.to(device, non_blocking=True)

        _, estimated, *_ = model(mixture, enrollment)

        # to numpy (1D)
        estimated_np = estimated.squeeze().detach().cpu().numpy()
        target_np = target.squeeze().detach().cpu().numpy()

        # metrics
        p = float(pesq(sr, target_np, estimated_np, mode="nb"))
        e = float(stoi(target_np, estimated_np, sr, extended=True))
        s = float(calculate_sisdr(target_np, estimated_np))

        pesq_list.append(p)
        estoi_list.append(e)
        sisdr_list.append(s)

        # optional audio saving
        if save_audio:
            out = rms_match(estimated_np, target_np) if rms_normal else estimated_np
            name = dataset._get_target_audio_file_name(idx)
            audio_path = os.path.join(recon_path, f"{name}")
            sf.write(audio_path, out, sr)

    pesq_arr = np.asarray(pesq_list)
    estoi_arr = np.asarray(estoi_list)
    sisdr_arr = np.asarray(sisdr_list)

    results = {
        "pesq_mean": float(pesq_arr.mean()),
        "estoi_mean": float(estoi_arr.mean()),
        "sisdr_mean": float(sisdr_arr.mean()),
    }

    logger.info("---------------------------------- RESULTS --------------------------------------")
    logger.info("AVG -- PESQ: %.4f, eSTOI: %.4f, SI-SDR: %.4f",
                results["pesq_mean"], results["estoi_mean"], results["sisdr_mean"])
    logger.info("---------------------------------------------------------------------------------")

    print("\n---------------------------------- RESULTS --------------------------------------")
    print(f"AVG -- PESQ: {results['pesq_mean']:.4f}, eSTOI: {results['estoi_mean']:.4f}, SI-SDR: {results['sisdr_mean']:.4f}")
    print("---------------------------------------------------------------------------------\n")

    return results


def main():
    parser = build_argparser()
    raw = parser.parse_args()

    # env first
    os.environ["CUDA_VISIBLE_DEVICES"] = raw.cuda_visible_devices

    args = Args(
        seed=raw.seed,
        train_lengths=raw.train_lengths,
        reference_lengths=raw.reference_lengths,
        fusion_type=raw.fusion_type,
        temperature=raw.temperature,
        R_coarse=raw.R_coarse,
        R_fine=raw.R_fine,
        M=raw.M,
        H=raw.H,
        compression_factor=raw.compression_factor,
        sr=raw.sr,
        Libri2Mix_path=raw.Libri2Mix_path,
        Libri2Mix_test_list=raw.Libri2Mix_test_list,
        save_path=raw.save_path,
        converged_model=raw.converged_model,
        log_path=raw.log_path,
        recon_path=raw.recon_path,
        n_fft=raw.n_fft,
        window_length=raw.window_length,
        hop_size=raw.hop_size,
        cuda_visible_devices=raw.cuda_visible_devices,
        rms_normal=raw.rms_normal,
        inference_task=raw.inference_task,
        save_audio=raw.save_audio,
    )

    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is unavailable. Please check CUDA installation.")

    device = torch.device("cuda")
    run_name = f"TEST_{get_timestamp()}_{os.path.splitext(args.converged_model)[0]}"
    logger = setup_logger(args.log_path, run_name)

    logger.info("Inference config: %s", args)

    dataset = build_dataset(args)
    model = load_model(args, device)

    n_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Test samples: %d, Params: %.2fM", len(dataset), n_params_m)
    logger.info("Recon path: %s", args.recon_path)
    logger.info("fusion=%s, n_fft=%d, win=%d, hop=%d", args.fusion_type, args.n_fft, args.window_length, args.hop_size)

    run_inference(
        model=model,
        dataset=dataset,
        device=device,
        logger=logger,
        recon_path=args.recon_path,
        sr=args.sr,
        rms_normal=args.rms_normal,
        save_audio=args.save_audio,
    )


if __name__ == "__main__":
    main()
