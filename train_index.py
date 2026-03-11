import os
import faiss
import numpy as np
import traceback
import argparse
import platform
from sklearn.cluster import MiniBatchKMeans


def train_index(exp_dir1, version19):
    n_cpu = os.cpu_count()
    exp_dir = "logs/%s" % (exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if not os.path.exists(feature_dir):
        print("Feature extraction not done! Make sure you extract features first.")
        return

    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        print("Feature extraction empty! Make sure you extract features first.")
        return

    print(f"Loading {len(listdir_res)} features...")
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)

    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]

    if big_npy.shape[0] > 2e5:
        print(f"Trying doing kmeans {big_npy.shape[0]} shape to 10k centers.")
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256
                    * n_cpu,  # Adjust batch size if needed (was 256 * config.n_cpu)
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except Exception as e:
            print(f"Error during K-Means: {traceback.format_exc()}")

    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    print(f"Shape: {big_npy.shape}, n_ivf: {n_ivf}")

    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    print("Training the index...")

    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)

    trained_index_path = "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index" % (
        exp_dir,
        n_ivf,
        index_ivf.nprobe,
        exp_dir1,
        version19,
    )
    faiss.write_index(index, trained_index_path)
    print(f"Done training index, saved to: {trained_index_path}")

    print("Adding features to index...")
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])

    added_index_path = "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index" % (
        exp_dir,
        n_ivf,
        index_ivf.nprobe,
        exp_dir1,
        version19,
    )
    faiss.write_index(index, added_index_path)
    print(f"Successfully built index: {added_index_path}")

    # Copy to weights folder exactly like original behavior
    outside_index_root = os.getenv("outside_index_root", "assets/indices")
    os.makedirs(outside_index_root, exist_ok=True)
    try:
        link = os.link if platform.system() == "Windows" else os.symlink

        target_path = "%s/%s_IVF%s_Flat_nprobe_%s_%s_%s.index" % (
            outside_index_root,
            exp_dir1,
            n_ivf,
            index_ivf.nprobe,
            exp_dir1,
            version19,
        )
        if os.path.exists(target_path):
            os.remove(target_path)

        link(added_index_path, target_path)
        print(f"Linked index to outside directory: {outside_index_root}")
    except Exception as e:
        print(f"Failed to link index to outside directory: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--experiment_dir", type=str, required=True, help="Experiment name"
    )
    parser.add_argument(
        "-v", "--version", type=str, default="v2", help="Version (v1 or v2)"
    )
    args = parser.parse_args()

    train_index(args.experiment_dir, args.version)
