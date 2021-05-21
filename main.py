import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


img = Image.open('Image.jpg')

red_band = img.getdata(band=0)
# convert to numpy array
img_mat = np.array(list(red_band), float)
# get image shape
img_mat.shape = (img.size[1], img.size[0])
# conver to 1d-array to matrix
img_mat = np.matrix(img_mat)

plt.imshow(img_mat)

img_mat_scaled = (img_mat - img_mat.mean()) / img_mat.std()
U, s, V = np.linalg.svd(img_mat_scaled)
num_components = 5
reconst_img_5 = np.matrix(U[:, :num_components]) * np.diag(s[:num_components]) * np.matrix(V[:num_components, :])

plt.plot(reconst_img_5)
plt.savefig('comp2.png', bbox_inches='tight', dpi=150)


fig, axs = plt.subplots(1, 2, figsize=(10, 10))
axs[0].imshow(img)
axs[0].set_title('Original Image', size=16)
axs[1].imshow(img_mat)
axs[1].set_title(' "R" band image', size=16)
plt.tight_layout()
plt.savefig('all.jpg', dpi=150)
print(type(img_mat))

img_mat_scaled = (img_mat - img_mat.mean()) / img_mat.std()
U, s, V = np.linalg.svd(img_mat_scaled)
var_explained = np.round(s ** 2 / np.sum(s ** 2), decimals=3)

plt.xlabel('Singular Vector', fontsize=16)
plt.ylabel('Variance Explained', fontsize=16)
plt.tight_layout()
plt.savefig('svd_scree_plot.png', dpi=150)

num_components = 5
reconst_img_5 = np.matrix(U[:, :num_components]) * np.diag(s[:num_components]) * np.matrix(V[:num_components, :])
plt.imshow(reconst_img_5)
plt.title('Reconstructed Image: 5 SVs', size=16)
plt.savefig('reconstructed_image_with_5_SVs.png', dpi=150)

num_components = 1000
reconst_img_1000 = np.matrix(U[:, :num_components]) * np.diag(s[:num_components]) * np.matrix(V[:num_components, :])
plt.imshow(reconst_img_1000)
plt.title('Reconstructed Image: 1000 SVs', size=16)
plt.savefig('reconstructed_image_with_1000_SVs.png', dpi=150)

fig, axs = plt.subplots(3, 3, figsize=(10, 10))
axs[0, 0].imshow(reconst_img_5)
axs[0, 0].set_title('Reconstructed Image: 5 SVs', size=16)
axs[1, 2].imshow(reconst_img_1000)
axs[1, 2].set_title('Reconstructed Image: 1000 SVs', size=16)
plt.tight_layout()
plt.savefig('reconstructed_images_using_different_SVs.jpg', dpi=150)
