import matplotlib.pyplot as plt
import numpy as np
import cv2

def cv2_imshow(image):
    # developed by Kanishke Gamagedara, udpated by MAE6292
    plt.figure(dpi=200)
    mode = len(np.shape(image))
    if mode==3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif mode==2:
        plt.imshow(image, cmap='gray')
    else:
        print('Unsuported image size')
    plt.xticks([]), plt.yticks([])
    plt.axis('off')

