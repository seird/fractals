import glob
import os


if __name__ == "__main__":
    for file in glob.glob("./example_*.py"):
        print(file)
        os.system(f"python {file}")
