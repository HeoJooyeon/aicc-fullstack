from pathlib import Path
import shutil

base_path = Path("C:\\Users\\MSI\\Documents\\test")

folder_mapping = {
    "images" : [".jpg",".jpeg",".png",".gif"],
    "texts" : [".txt"],
    "hwp_files" : [".hwp"]
}

for key in folder_mapping:
    full_path = base_path / key
    if full_path.exists() and full_path.is_dir():
        for file in full_path.iterdir():
            if file.is_file():
                shutil.move(str(file), str(base_path))
                print(f"Moved {file.name} -> {base_path}/")
        full_path.rmdir()