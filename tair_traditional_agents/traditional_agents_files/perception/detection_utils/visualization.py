import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image as im
cams = ['front_left', 'front', 'front_right', 'back_right', 'back', 'back_left']
others = ['perception_vehicle', 'perception_walker', 'roach_input']

def plot_images(images, path, count, monocular):
    resize_shape = (200,200)
    if not monocular:
        fig, axs = plt.subplots(3, 3)
        for i in range(6):
            img = images['cameras'][cams[i]][1][:, :, -2::-1]
            img = im.fromarray(img)
            img = img.resize(resize_shape)
            axs[i // 3, i % 3].grid(False)
            axs[i // 3, i % 3].imshow(img)  # Transpose the image to
            axs[i // 3, i % 3].set_title(cams[i])
            #axs[i // 3, i % 3].set_aspect('auto')

        for idx, k in enumerate(others):
            img = images[k]
            if idx < 2:
                img = im.fromarray(img[:, :, 0])
            else:
                img = im.fromarray(img)
            img = img.resize(resize_shape)
            if img is not None:
                axs[2, idx % 3].imshow(img)
                axs[2, idx % 3].set_title(others[idx])
                axs[2, idx % 3].grid(False)
                #axs[2, idx % 3].set_aspect('auto')

        #plt.tight_layout()
        plt.savefig(path + str(count)+ '.png')
        plt.cla()
        plt.close(fig)
        return

    else:
        fig, axs = plt.subplots(1, 4)

        img = images['cameras']['front'][1][:, :, -2::-1]
        img = im.fromarray(img)
        img = img.resize(resize_shape)
        axs[0].grid(False)
        axs[0].imshow(img)  # Transpose the image to
        axs[0].set_title('front')
        #axs[0].set_aspect('auto')

        for idx, k in enumerate(others):
            img = images[k]
            if idx < 2:
                img = im.fromarray(img[:, :, 0])
            else:
                img = im.fromarray(img)
            img = img.resize(resize_shape)
            if img is not None:
                axs[idx + 1].imshow(img)
                axs[idx + 1].set_title(others[idx])
                axs[idx + 1].grid(False)
                #axs[idx + 1].set_aspect('auto')

        plt.tight_layout()
        plt.savefig(path + str(count) + '.png')
        plt.cla()
        plt.close(fig)
        return