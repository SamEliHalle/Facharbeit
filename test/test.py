import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
from skimage import transform
from skimage.color import gray2rgb
from keras.callbacks import Callback
from matplotlib import pyplot as plt
import os

def alpha_blend(rgba, rgb): # von Copilot
  alpha = rgba[3] / 255.0
  blended = [alpha * rgba[i] + (1 - alpha) * rgb[i] for i in range(3)]
  return list(map(int, blended))

def mergeImageMask(image_suffix="", mask_suffix="_predict", merge_suffix="_overlay", save_path="data/cells/test"):
    for i, filename in enumerate(os.listdir(save_path)):
      if filename.endswith(mask_suffix + ".png"):
        continue
      
      img = io.imread(os.path.join(save_path, filename), as_gray=False)
      img = gray2rgb(img)

      maskname = ""
      merged = ""
      if image_suffix != "":
        maskname = filename.replace(image_suffix, mask_suffix)
        merged = filename.replace(image_suffix, merge_suffix)
      else:
        maskname = filename.split(".")[0] + mask_suffix + ".png"
        merged = filename.split(".")[0] + merge_suffix + ".png"
      
      mask = io.imread(os.path.join(save_path, maskname))
      
      for i in range(0, len(mask)):
        for j in range(0, len(mask[i])):
          if mask[i][j] < 100:
            alpha = 255 - mask[i][j]
            green = (34, 139, 34, alpha) # grün
            img[i][j] = alpha_blend(green, img[i][j])
      
      io.imsave(os.path.join(save_path, merged), img)

        