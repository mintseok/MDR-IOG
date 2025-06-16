import os
import shutil

# 기본 경로 설정
DATA_DIR = "data"
PT_DIR = "pt_data"
PT_TYPES = ["koclip_pt_end", "koclip_pt_start", "koclip_pt_image"]

def sync_tensor_files():
    moved_files = []

    # 모든 label 디렉토리 확인
    for new_label in os.listdir(DATA_DIR):
        new_label_path = os.path.join(DATA_DIR, new_label)
        if not os.path.isdir(new_label_path):
            continue

        # 해당 label 디렉토리의 파일 확인
        for site_file in os.listdir(new_label_path):
            full_path = os.path.join(new_label_path, site_file)
            if not os.path.isdir(full_path):
                continue

            # 실제 label 확인 (ex: 이전 경로가 무엇이었는지는 .pt 경로로 추정)
            for pt_type in PT_TYPES:
                pt_root = os.path.join(PT_DIR, pt_type)

                # 현재 .pt 파일이 저장되어 있는 폴더 중에서 이 사이트 이름을 가진 파일을 찾기
                for old_label in os.listdir(pt_root):
                    old_label_path = os.path.join(pt_root, old_label)
                    pt_file_name = f"{site_file}.pt"
                    pt_file_path = os.path.join(old_label_path, pt_file_name)

                    # 해당 pt 파일이 존재한다면 → 이동
                    if os.path.exists(pt_file_path):
                        new_pt_dir = os.path.join(pt_root, new_label)
                        os.makedirs(new_pt_dir, exist_ok=True)
                        new_pt_path = os.path.join(new_pt_dir, pt_file_name)

                        print(f"Moving: {pt_file_path} → {new_pt_path}")
                        shutil.move(pt_file_path, new_pt_path)
                        moved_files.append((pt_file_path, new_pt_path))
                        break  # 한 번만 이동하면 되므로 break

    print(f"\nMoved a total number of {len(moved_files)} .pt files.")

if __name__ == "__main__":
    sync_tensor_files()
