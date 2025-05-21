from pathlib import Path
import shutil

base_path = Path("C:\\Users\\MSI\\Documents\\test")

folder_mapping = {
    "images" : [".jpg",".jpeg",".png",".gif"],
    "texts" : [".txt"],
    "hwp_files" : [".hwp"]
}

for file in base_path.iterdir():
    if file.is_file():
        ext = file.suffix.lower()
        
        for folder_name, extentions in folder_mapping.items():
            if ext in extentions:
                target_folder = base_path / folder_name
                target_folder.mkdir(exist_ok=True)
                
                shutil.move(str(file), str(target_folder/file.name))
                print(f"Moved {file.name} -> {folder_name}/")
                break