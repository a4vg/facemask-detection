import cv2
import os
import numpy as np

def load_dataset( img_dirs, preprocessors=[], verbose=1 ):
  data = []
  labels = []
  for img_dir in img_dirs:
    img_paths = os.listdir(img_dir)

    for i, img_path in enumerate( img_paths ):
      if img_path == ".DS_Store": continue
      img_path = f"{img_dir}/{img_path}"
      img = cv2.imread( img_path )
      label = img_path.split(os.path.sep)[-2]
      for p in preprocessors:
        img = p(img)
      data.append( img )    
      labels.append( label )
      if verbose>0 and i>0 and (i+1)%verbose==0:
        print( f"Preprocessing {label} {i+1}/{len(img_paths)}" )
  return np.array(data), labels

def resize(w, h):
  return lambda img: cv2.resize( img, (w, h), cv2.INTER_AREA)

def binarize_labels(labels, classes):
  bin_labels = []
  for l in labels:
    vec = [0]*len(classes)
    vec[classes.index(l)] = 1
    bin_labels.append(vec)
  return np.array(bin_labels)

def preprocess_single(img, preprocessors):
  for p in preprocessors: img = p(img)
  return np.array([img])

'''
data, labels = load_dataset(
                ["dataset/Train/WithMask", "dataset/Train/WithoutMask"],
                [resize(50,50), img_to_array],
                1000
              )
print(data.shape, data[0].shape, labels[:10])
'''

