"""
download_data.py - download the HAI 21.03 dataset from the official repository.

Source: https://github.com/icsdataset/hai (hai-21.03)
Files are saved as-is (.csv.gz) to:
  data/train/train1.csv.gz ... train3.csv.gz
  data/test/test1.csv.gz  ... test5.csv.gz

Usage:
  cd code
  python download_data.py
"""

import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/icsdataset/hai/master/hai-21.03"
FILES = {
    "train": ["train1.csv.gz", "train2.csv.gz", "train3.csv.gz"],
    "test": ["test1.csv.gz", "test2.csv.gz", "test3.csv.gz", "test4.csv.gz", "test5.csv.gz"],
}
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    for split, names in FILES.items():
        out_dir = DATA_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            dest = out_dir / name
            if dest.exists():
                print(f"[skip] {dest} already exists")
                continue
            url = f"{BASE_URL}/{name}"
            print(f"[download] {url}")
            tmp = dest.with_name(dest.name + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)
    print(f"Done. Data saved under {DATA_DIR}")


if __name__ == "__main__":
    main()
