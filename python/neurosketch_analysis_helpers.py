import numpy as np
import nibabel as nib
import scipy.stats as stats

def get_vectorized_voxels_from_map(filename):
  img = nib.load(filename)
  data = img.get_data()
  flat = np.ravel(data)
  return flat

