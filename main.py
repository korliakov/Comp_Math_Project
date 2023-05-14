import numpy as np

from DataProcessing import DataProcessor, Preprocessing
from Model import Unet, FigsDataset, vis_dataset, init_model, unet_make_prediction, get_coords_from_mask, dbscan_make_prediction
from constants import *
import matplotlib.pyplot as plt



if __name__ == '__main__':

    preproc = Preprocessing()
    list_images = preproc.prepare("./test", SIZE)

    model = init_model(MODEL_PATH)

    x = np.arange(0, SIZE[0])
    y = np.arange(0, SIZE[1])

    xv,yv = np.meshgrid(x, y)

    for image in list_images:
        unet_pred = unet_make_prediction(image, model, THRESHOLD)
        coords = get_coords_from_mask(unet_pred, xv, yv)
        dbscan_pred = dbscan_make_prediction(coords, EPS, MIN_SAMPLES)

        print(np.unique(dbscan_pred))

        plt.imshow(unet_pred)
        plt.show()

        plt.scatter(coords.T[0], -coords.T[1], c=dbscan_pred, s=10)
        plt.show()



