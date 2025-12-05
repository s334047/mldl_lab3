import os
import urllib.request
import zipfile

URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
ZIP_PATH = "data/tiny-imagenet-200.zip"
DEST_FOLDER = "data"

def download_tiny_imagenet():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(ZIP_PATH):
        print("Downloading TinyImageNet...")
        urllib.request.urlretrieve(URL, ZIP_PATH)
        print("Download complete.")
    else:
        print("Zip already downloaded.")
    
    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DEST_FOLDER)
        
    print("Dataset extracted.")
    
def reorganize_val_set():
    """ Moves validation images into class folders based on val_annotations.txt
    """
    import shutil 
    
    val_dir = "data/tiny-imagenet-200/val"
    annotations = os.path.join(val_dir, "val_annotations.txt")
    
    with open(annotations) as f:
        for line in f:
            file, cls, *_ = line.split()
            class_dir = os.path.join(val_dir, cls)
            os.makedirs(class_dir, exist_ok=True)
            src = os.path.join(val_dir, "images", file)
            dst = os.path.join(class_dir, file)
            shutil.copyfile(src, dst)
    
    shutil.rmtree(os.path.join(val_dir, "images"))
    print("Validation set reorganized.")

if __name__ == "__main__":
    download_tiny_imagenet()
    reorganize_val_set()