import os

def count_data_by_label(data_dir="data"):
    print(f"📂 Data directory: {data_dir}")
    print("Data per label:\n")

    total = 0
    for label_name in sorted(os.listdir(data_dir)):
        label_path = os.path.join(data_dir, label_name)
        if os.path.isdir(label_path):
            subdirs = [d for d in os.listdir(label_path)
                       if os.path.isdir(os.path.join(label_path, d))]
            count = len(subdirs)
            total += count
            print(f"{label_name}: {count}")

    print(f"\n🔢 Total number of data: {total}")

if __name__ == "__main__":
    count_data_by_label()
