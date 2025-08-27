import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grad_norm_path", type=str, default="./checkpoints/grad_and_epsilon_batch8_no_train/grad_norms.npy")
    parser.add_argument("--security_grad_norm_path", type=str, default="./checkpoints/grad_and_epsilon_batch8_no_train/security_grad_norms.npy")
    parser.add_argument("--security_epsilon_path", type=str, default="./checkpoints/grad_and_epsilon_batch8_no_train/security_epsilons.npy")
    parser.add_argument("--output_dir", type=str, default="./test_results")
    args = parser.parse_args()

    grad_norms = np.load(args.grad_norm_path)
    security_grad_norms = np.load(args.security_grad_norm_path)
    security_epsilons = np.load(args.security_epsilon_path)
    os.makedirs(args.output_dir, exist_ok=True)

    # grad distribution
    plt.figure()
    plt.hist(grad_norms, bins=50, density=True, alpha=0.5, label="normal")
    plt.hist(security_grad_norms, bins=50, density=True, alpha=0.5, label="security_norm")
    plt.xlabel("Gradient L2 Norm")
    plt.ylabel("Density")
    plt.title("Gradient Norm Distribution")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "gradient_norm_distribution.png"))
    plt.clf()

    # grad - training_steps
    plt.figure()
    plt.plot(grad_norms)
    plt.xlabel("Training Steps")
    plt.ylabel("Gradient L2 Norm")
    plt.title("Gradient Norm over training")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "gradient_norm_over_steps.png"))
    plt.clf()

    # security grad - training_steps
    plt.figure()
    plt.plot(security_grad_norms)
    plt.xlabel("Training Steps")
    plt.ylabel("Security Gradient L2 Norm")
    plt.title("Security Gradient Norm over training")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "security_gradient_norm_over_steps.png"))
    plt.clf()

    # epsilon - training_steps
    plt.figure()
    plt.plot(security_epsilons)
    plt.xlabel("Training Steps")
    plt.ylabel("Privacy Epsilon")
    plt.title("Privacy budget over training")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "security_epsilon_over_steps.png"))
    plt.clf()
    