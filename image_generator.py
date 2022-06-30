import cv2
import os
import numpy as np
import imageio

class ImageGenerator():
    def __init__(self, dataset_path, GAN_type = "P2P", img_dimensions = (128, 128)):
        self.GAN_type = GAN_type
        self.dataset_path = dataset_path
        self.img_dimensions = img_dimensions
    def generate_batch_images(self, batch_size=1, is_testing=False):

        image_files = os.listdir(os.path.join(self.dataset_path, "train" if not is_testing else "val"))

        self.n_batches = int(len(image_files) / batch_size)

        for i in range(self.n_batches-1):
            batch = image_files[i*batch_size:(i+1)*batch_size]
            list_groundtruth_img, list_condition_img = [], []
            for img_name in batch:
                img = self.imread(os.path.join(self.dataset_path, "train" if not is_testing else "test", img_name))
                h, w, _ = img.shape
                half_width = int(w/2)
                groundtruth_img = img[:, :half_width, :]
                condition_img = img[:, half_width:, :]

                groundtruth_img = cv2.resize(groundtruth_img, self.img_dimensions)
                condition_img = cv2.resize(condition_img, self.img_dimensions)

                if not is_testing and np.random.random() > 0.5:
                        groundtruth_img = np.fliplr(groundtruth_img)
                        condition_img = np.fliplr(condition_img)

                list_groundtruth_img.append(groundtruth_img)
                list_condition_img.append(condition_img)

            list_groundtruth_img = np.array(list_groundtruth_img)/127.5 - 1.
            list_condition_img = np.array(list_condition_img)/127.5 - 1.

            if (self.GAN_type == "P2P"):
                yield list_groundtruth_img, list_condition_img
            else:
                yield list_groundtruth_img

    def generate_img_data(self, batch_size = 1, is_testing = False, ):
        image_files = os.listdir(os.path.join(self.dataset_path, "train" if not is_testing else "test"))

        list_groundtruth_img = []
        list_condition_img = []

        batch_images = np.random.choice(image_files, size=batch_size)
        for img_name in batch_images:
            img = self.imread(os.path.join(self.dataset_path, "train" if not is_testing else "test", img_name))
            h, w, _ = img.shape
            half_width = int(w/2)
            groundtruth_img, condition_img = img[:, :half_width, :], img[:, half_width:, :]
            groundtruth_img = cv2.resize(groundtruth_img, self.img_dimensions)
            condition_img = cv2.resize(condition_img, self.img_dimensions)


            if not is_testing and np.random.random() < 0.5:
                groundtruth_img = np.fliplr(groundtruth_img)
                condition_img = np.fliplr(condition_img)

            list_groundtruth_img.append(groundtruth_img)
            list_condition_img.append(condition_img)

        list_groundtruth_img = np.array(list_groundtruth_img)/127.5 - 1.
        list_condition_img = np.array(list_condition_img)/127.5 - 1.

        if(self.GAN_type == "P2P"):
            return list_groundtruth_img, list_condition_img
        else:
            return(list_groundtruth_img)


    def imread(self, path):
        return imageio.imread(path, pilmode="RGB").astype(np.float)
