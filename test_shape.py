import numpy as np

path = r"D:\College notes\SEM-6\Minor Project\NSL\Data Collection\Data\raw\building_on_fire\building_on_fire__signer02__001.npy"

data = np.load(path)

print("Shape  :", data.shape)
print("Dtype  :", data.dtype)
print("First row of data:")
print(data[0])