import numpy as np

def np2carray(arr):
  if np.isscalar(arr):
    return str(arr)
  else:
    return "{ " + ", ".join(map(np2carray, arr)) + " }"
