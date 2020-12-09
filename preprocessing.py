from tensorflow import keras
import argparse
import os
import pickle
import numpy as np

def load_data(path):
    with np.load(path) as f:
        train_images, train_labels = f['x_train'], f['y_train']
        test_images, test_labels = f['x_test'], f['y_test']
        return (train_images, train_labels), (test_images, test_labels)

def preprocess(data_dir: str):
    #fashion_mnist = keras.datasets.fashion_mnist
    #(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    (train_images, train_labels), (test_images, test_labels) = load_data(path="/data/mnist.npz")
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(data_dir, 'train_images.pickle'), 'wb') as f:
        pickle.dump(train_images, f)

    with open(os.path.join(data_dir, 'train_labels.pickle'), 'wb') as f:
        pickle.dump(train_labels, f)

    with open(os.path.join(data_dir, 'test_images.pickle'), 'wb') as f:
        pickle.dump(test_images, f)

    with open(os.path.join(data_dir, 'test_labels.pickle'), 'wb') as f:
        pickle.dump(test_labels, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kubeflow MNIST training script')
    parser.add_argument('--data_dir', help='path to images and labels.')
    args = parser.parse_args()

    preprocess(data_dir=args.data_dir)
