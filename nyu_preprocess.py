import os
from sklearn.model_selection import train_test_split


def preprocess_nyu_dataset(base_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mapping = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith("rgb_") and file.endswith(".jpg"):
                rgb_path = os.path.join(root, file)
                depth_file = file.replace("rgb_", "sync_depth_").replace(".jpg", ".png")
                depth_path = os.path.join(root, depth_file)

                if os.path.exists(depth_path):
                    mapping.append(f"{rgb_path} {depth_path}")

    full_mapping_file = os.path.join(output_dir, "full_mapping.txt")
    with open(full_mapping_file, "w") as f:
        f.write("\n".join(mapping))

    train, test = train_test_split(mapping, test_size=0.2, random_state=42)

    train_file = os.path.join(output_dir, "train.txt")
    test_file = os.path.join(output_dir, "test.txt")

    with open(train_file, "w") as f:
        f.write("\n".join(train))

    with open(test_file, "w") as f:
        f.write("\n".join(test))

    print(f"Preprocessing complete. Files saved in {output_dir}:")
    print(f"- Full mapping: {full_mapping_file}")
    print(f"- Train split: {train_file}")
    print(f"- Test split: {test_file}")


base_dir = "/path/to/nyu/sync"
output_dir = "/path/to/output"
preprocess_nyu_dataset(base_dir, output_dir)

