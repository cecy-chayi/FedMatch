import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_tasks(base_dir):
    files = [f for f in os.listdir(base_dir) if f.endswith('.npy')]
    data_by_file = {}
    for f in files:
        path = os.path.join(base_dir, f)
        data = np.load(path, allow_pickle=True).item()
        data_by_file[f] = data
    return data_by_file

def plot_label_distributions(data_by_file, save_dir=None):
    s_files = [k for k in data_by_file if k.startswith('s_')]
    u_files = [k for k in data_by_file if k.startswith('u_')]

    def plot(files, title, fig_name):
        plt.figure(figsize=(10, 6))
        for fname in sorted(files):
            data = data_by_file[fname]
            labels = np.argmax(data['y'], axis=1)
            unique, counts = np.unique(labels, return_counts=True)
            plt.bar([f"{fname[:-4]}_{int(l)}" for l in unique], counts, label=fname[:-4])

        plt.xticks(rotation=90)
        plt.title(title)
        plt.xlabel('Client_Label')
        plt.ylabel('Sample Count')
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, fig_name))
        else:
            plt.show()
        plt.close()

    plot(s_files, "Labeled Data Distribution", "labeled_distribution.png")
    plot(u_files, "Unlabeled Data Distribution", "unlabeled_distribution.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Path to saved .npy data files")
    parser.add_argument('--save_dir', type=str, default=None, help="Optional path to save plots")
    args = parser.parse_args()

    data_by_file = load_tasks(args.data_dir)
    plot_label_distributions(data_by_file, args.save_dir)
