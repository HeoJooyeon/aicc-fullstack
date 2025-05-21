from pathlib import Path

path = Path("exam.txt")

try:
    contents = path.read_text()
    print(contents)
except FileNotFoundError as e:
    print(f"file not found: {e}")