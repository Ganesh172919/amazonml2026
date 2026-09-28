#!/usr/bin/env python3
"""
Master runner script for Business Entity Resolution.
Trains the matching model, generates candidates and final predictions, and validates the output.
"""

import argparse
import os
import sys
import subprocess

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import train_pipeline, inference_pipeline
from src.pipeline_v2 import train_pipeline_v2, inference_pipeline_v2


def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution End-to-End Pipeline")
    parser.add_argument(
        "--version",
        type=int,
        choices=[1, 2],
        default=1,
        help="Pipeline version: 1 (Multi-key Lexical + RapidFuzz + LightGBM) or 2 (TF-IDF + RapidFuzz + BGE + LightGBM)",
    )
    parser.add_argument(
        "--data-dir",
        default="student_resource/dataset",
        help="Path to dataset root folder containing train/ and test/ folders",
    )
    parser.add_argument(
        "--output-dir",
        default="student_resource/output",
        help="Path to folder where matching_results.tsv and candidate_pairs.tsv will be saved",
    )
    parser.add_argument(
        "--model-path",
        default="models/matching_lgbm.pkl",
        help="Path to save / load trained model checkpoint",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=50000,
        help="Number of training entities to sample for training",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training if a trained model already exists at --model-path",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Run utils/validate_submission.py after inference",
    )

    args = parser.parse_args()

    # Step 1: Train model
    if os.path.exists(args.model_path) and args.skip_train:
        print(f"Found existing model at {args.model_path}. Skipping training.")
    else:
        print(f"Starting training phase (Version {args.version})...")
        if args.version == 2:
            train_pipeline_v2(
                data_dir=args.data_dir,
                model_save_path=args.model_path,
                sample_entities=args.train_samples,
            )
        else:
            train_pipeline(
                data_dir=args.data_dir,
                model_save_path=args.model_path,
                sample_entities=args.train_samples,
            )

    # Step 2: Inference
    print(f"\nStarting test inference phase (Version {args.version})...")
    if args.version == 2:
        inference_pipeline_v2(
            data_dir=args.data_dir,
            model_path=args.model_path,
            output_dir=args.output_dir,
        )
    else:
        inference_pipeline(
            data_dir=args.data_dir,
            model_path=args.model_path,
            output_dir=args.output_dir,
        )

    # Step 3: Validate
    if args.validate:
        validator_path = "student_resource/utils/validate_submission.py"
        test_dir = os.path.join(args.data_dir, "test")
        matching_file = os.path.join(args.output_dir, "matching_results.tsv")
        candidate_file = os.path.join(args.output_dir, "candidate_pairs.tsv")

        if os.path.exists(validator_path):
            print("\n=======================================================")
            print("Running official submission validator...")
            print("=======================================================")
            cmd = [
                sys.executable,
                validator_path,
                "--matching", matching_file,
                "--candidate", candidate_file,
                "--test-dir", test_dir,
            ]
            res = subprocess.run(cmd)
            if res.returncode == 0:
                print("\nSUCCESS: All files passed official validation!")
            else:
                print("\nWARNING: Validator reported issues. Check above log.")
        else:
            print(f"\nValidator script not found at {validator_path}. Skipping validator check.")


if __name__ == "__main__":
    main()
