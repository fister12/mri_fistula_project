from __future__ import annotations

import argparse
from pathlib import Path

from preprocessing import preprocess_patient
from train import run_training


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Run MRI preprocessing only or the full preprocessing + training pipeline."
	)
	parser.add_argument(
		"--mode",
		choices=("preprocess", "train", "all"),
		default="preprocess",
		help="Choose how far to run the pipeline.",
	)
	parser.add_argument(
		"--input-dicom",
		type=Path,
		required=True,
		help="Path to the DICOM folder for preprocessing.",
	)
	parser.add_argument(
		"--preprocessed-output",
		type=Path,
		default=Path("data/processed"),
		help="Where preprocessed volumes will be written.",
	)
	parser.add_argument(
		"--training-data",
		type=Path,
		default=Path("data/processed"),
		help="Path to preprocessed data used for training.",
	)
	parser.add_argument(
		"--model-output",
		type=Path,
		default=Path("models"),
		help="Directory where the trained model will be saved.",
	)
	return parser


def main() -> None:
	args = build_parser().parse_args()

	if args.mode in {"preprocess", "all"}:
		preprocess_patient(args.input_dicom, args.preprocessed_output)

	if args.mode in {"train", "all"}:
		run_training(args.training_data, args.model_output)


if __name__ == "__main__":
	main()
