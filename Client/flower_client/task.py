"""tfflower: A Flower / TensorFlow app."""

import os
import pandas as pd

import keras
from keras import layers
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from matplotlib import pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
from keras._tf_keras.keras.utils import load_img, img_to_array
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


'''def load_model(learning_rate: float = 0.001):
    model = keras.Sequential([
    layers.Conv2D(32, (5,5), padding='same', activation='relu', input_shape=(28,28,1)),
    layers.Conv2D(32, (5,5), padding='same', activation='relu'),
    layers.MaxPool2D(),
    layers.Dropout(0.25),
    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.MaxPool2D(strides=(2,2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
    ])
    optimizer = keras.optimizers.Adam(learning_rate)

    model.compile(optimizer=optimizer, 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
              metrics=['acc'])
    return model


fds = None  # Cache FederatedDataset'''

def load_model(learning_rate: float = 0.001):
    model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(139,)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5, activation='softmax')  
])
    optimizer = keras.optimizers.Adam(learning_rate)

# Compile 
    model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=["accuracy"]
    )
    return model


'''def load_data(partition_id, num_partitions):
    # Download and partition dataset
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="label",alpha=1.0)
        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id, "train")
    partition.set_format("numpy")

    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2)
    x_train, y_train = partition["train"]["image"] / 255.0, partition["train"]["label"]
    x_test, y_test = partition["test"]["image"] / 255.0, partition["test"]["label"]
    return x_train, y_train, x_test, y_test '''
def load_data(partition_id, num_partitions):
    data_files = "D:/TaiLieu/2024-2025/DoAn/Flower/tfflower/tfflower/feature_vectors_syscalls_frequency_5_Cat.csv"
    dataset_dict = load_dataset("csv", data_files=data_files)
    dataset = dataset_dict["train"]

    partitioner = IidPartitioner(num_partitions=num_partitions)
    partitioner.dataset = dataset
    partition = partitioner.load_partition(partition_id)

    partition_df = partition.to_pandas()

    # Convert to NumPy arrays
    partition_np = partition_df.to_numpy()

    # Divide data on each node: 80% train, 20% test
    X = partition_np[:, :-1]  # All columns except the last one
    y = partition_np[:, -1]   # Last column as labels
    y = y.reshape(-1, 1) 
    # One-hot encode the labels
    encoder = OneHotEncoder(sparse_output=False, categories=[np.array([1.0, 2.0, 3.0, 4.0, 5.0])])
    y = encoder.fit_transform(y)
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    

    return X_train, y_train, X_test, y_test



'''def load_data(partition_id: int, num_partitions: int):
    """Load the data for the given partition."""
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="hitorilabs/iris", partitioners={"train": partitioner}
        )
    dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]
    X = dataset[FEATURES]
    y = dataset["species"]
    # Split the on-edge data: 80% train, 20% test
    X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)) :]
    y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)) :]
    return X_train, y_train, X_test, y_test'''
'''    data_files = "D:/TaiLieu/2024-2025/DoAn/Flower/tfflower/tfflower/feature_vectors_syscalls_frequency_5_Cat.csv"
    dataset_dict = load_dataset("csv", data_files=data_files)
    dataset = dataset_dict["train"]

    partitioner = IidPartitioner(num_partitions=num_partitions)
    partitioner.dataset = dataset
    partition = partitioner.load_partition(partition_id)
    # Convert to numpy arrays
    partition_df = partition.to_pandas()

    # Convert to NumPy arrays
    partition_np = partition_df.to_numpy()
    # Divide data on each node: 80% train, 20% test
    X = partition_np[:, :-1]  # All columns except the last one
    y = partition_np[:, -1]   # Last column as labels
    y = keras.utils.to_categorical(y, num_classes=5)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test
'''