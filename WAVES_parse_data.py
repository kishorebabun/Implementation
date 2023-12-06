import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

#Paste the paths where the meshes and images are present in DATA_DIR and DATA_DIR2 respectively.

DATA_DIR = "lungpointcloudtrianglebrightness/lungpointcloud"
DATA_DIR2="lungpreprocessed-224/lung"

def parse_dataset(num_points=2048):

    train_points = []
    train_images=[]
    train_labels = []
    test_points = []
    test_images=[]
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(DATA_DIR, "*"))

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]
        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        tr_fil=os.path.join(folder,"train/")
        te_fil=os.path.join(folder,"test/")
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            print(f)
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)
            file=f.replace(DATA_DIR,"")
            path=DATA_DIR2+file[:-3:]+"jpg"
            img=Image.open(path)
            train_images.append(tf. convert_to_tensor(img))

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)
            file=f.replace(DATA_DIR,"")
            path=DATA_DIR2+file[:-3:]+"jpg"
            img=Image.open(path)
            test_images.append(tf. convert_to_tensor(img))

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_images),
        np.array(test_images),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )