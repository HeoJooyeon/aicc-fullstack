from pathlib import Path

path = Path("./example.txt")

path.write_text("tt")

contents = path.read_text()

print(contents.lower())