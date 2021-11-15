# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower client example using TensorFlow for Fashion-MNIST image classification."""


from typing import Tuple, List, cast

import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize



SEED = 2020


def load_model(input_shape: Tuple[int, int, int] = (28, 28, 1)) -> tf.keras.Model:
    """Load model for Fashion-MNIST."""
    # Kernel initializer
    kernel_initializer = tf.keras.initializers.glorot_uniform(seed=SEED)

    # Architecture
    inputs = tf.keras.layers.Input(shape=input_shape)
    layers = tf.keras.layers.Conv2D(
        32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer=kernel_initializer,
        padding="same",
        activation="relu",
    )(inputs)
    layers = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(layers)
    layers = tf.keras.layers.Conv2D(
        64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer=kernel_initializer,
        padding="same",
        activation="relu",
    )(layers)
    layers = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(layers)
    layers = tf.keras.layers.Flatten()(layers)
    layers = tf.keras.layers.Dense(
        512, kernel_initializer=kernel_initializer, activation="relu"
    )(layers)

    outputs = tf.keras.layers.Dense(
        10, kernel_initializer=kernel_initializer, activation="softmax"
    )(layers)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=["accuracy"],
    )
    return model


def load_data(
    partition: int, num_partitions: int
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load partition of randomly shuffled Fashion-MNIST subset."""
    # Load training and test data (ignoring the test data for now)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Take a subset
    x_train, y_train = shuffle(x_train, y_train, seed=SEED)
    x_test, y_test = shuffle(x_test, y_test, seed=SEED)

    x_train, y_train = get_partition_v2(x_train, y_train, partition, num_partitions)
    #x_test, y_test = get_partition(x_test, y_test, partition, num_partitions)

    #print('x_train shape: ', x_train.shape)
    #print('y_train shape: ', y_train.shape)
    #print('x_test shape: ', x_test.shape)
    #print('y_test shape: ', y_test.shape)
    
    # Adjust x sets shape for model
    x_train = adjust_x_shape(x_train)
    x_test = adjust_x_shape(x_test)

    # Normalize data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Convert class vectors to one-hot encoded labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def load_data_v2(
    num_of_edges: int, nonidd: bool, equal: bool, ratio: List[float], seed: int, log_path: str
) -> List[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    """Load partition of randomly shuffled Fashion-MNIST subset."""
    # Load training and test data (ignoring the test data for now)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Take a subset
    x_train, y_train = shuffle(x_train, y_train, seed=seed)
    x_test, y_test = shuffle(x_test, y_test, seed=seed)
    x_test = adjust_x_shape(x_test)
    x_test = x_test.astype("float32") / 255.0
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    dslists = []
    if equal:
        if nonidd:
            print('Split non-idd data')
            num_sanples = 1300
            for edge in range(num_of_edges):
                edge_x_train, edge_y_train = get_partition_v2(x_train, y_train, edge, num_sanples)
                # Adjust x sets shape for model
                edge_x_train = adjust_x_shape(edge_x_train)
                # Normalize data
                edge_x_train = edge_x_train.astype("float32") / 255.0
                # Convert class vectors to one-hot encoded labels
                #print('y_test shape: ', edge_y_train.shape)
                #print(edge_y_train[0:5])
                #labels = edge_y_train.tolist()
                #label_cnt = [ labels.count(label) for label in range(10)]
                #print(label_cnt)
                findall = np.where((edge_y_train == edge*2) | (edge_y_train == (edge*2)+1))
                deletes = random.sample(findall[0].tolist(), int(0.9*len(findall[0])))
                #print('delete ', len(deletes))
                edge_x_train = np.delete(edge_x_train,deletes,0)
                edge_y_train = np.delete(edge_y_train,deletes,0)
                
                labels = edge_y_train.tolist()
                label_cnt = [ labels.count(label) for label in range(10)]
                #print(label_cnt)
                edge_y_train = tf.keras.utils.to_categorical(edge_y_train, 10)
                #print('x_test shape: ', edge_x_train.shape)
                #print('y_test shape: ', edge_y_train.shape)
                dslists.append(((edge_x_train, edge_y_train), (x_test, y_test)))

                
                fig, ax = plt.subplots(1, 1)
                # Get a color map
                my_cmap = cm.get_cmap('jet')
                # Get normalize function (takes data in range [vmin, vmax] -> [0, 1])
                my_norm = Normalize(vmin=0, vmax=10)
                ax.bar(range(1,11), label_cnt, color=my_cmap(my_norm(range(1,11))))
                plt.tight_layout()
                plt.savefig(log_path+r'/data/edge_{}.png'.format(edge))
                
        else:
            #num_sanples = len(x_train)/num_of_edges
            num_sanples = 1000
            for edge in range(num_of_edges):
                edge_x_train, edge_y_train = get_partition_v2(x_train, y_train, edge, num_sanples)
                # Adjust x sets shape for model
                edge_x_train = adjust_x_shape(edge_x_train)
                # Normalize data
                edge_x_train = edge_x_train.astype("float32") / 255.0
                # Convert class vectors to one-hot encoded labels
                #print('y_test shape: ', edge_y_train.shape)
                #print(edge_y_train[0:5])
                labels = edge_y_train.tolist()
                label_cnt = [ labels.count(label) for label in range(10)]
                #print(label_cnt)
                edge_y_train = tf.keras.utils.to_categorical(edge_y_train, 10)
                dslists.append(((edge_x_train, edge_y_train), (x_test, y_test)))
                fig, ax = plt.subplots(1, 1)
                # Get a color map
                my_cmap = cm.get_cmap('jet')
                # Get normalize function (takes data in range [vmin, vmax] -> [0, 1])
                my_norm = Normalize(vmin=0, vmax=10)
                ax.bar(range(1,11), label_cnt, color=my_cmap(my_norm(range(1,11))))
                plt.tight_layout()
                plt.savefig(log_path+r'/data/edge_{}.png'.format(edge))
    else:
        print('Split non-equal data')

    return dslists


def adjust_x_shape(nda: np.ndarray) -> np.ndarray:
    """Turn shape (x, y, z) into (x, y, z, 1)."""
    nda_adjusted = np.reshape(nda, (nda.shape[0], nda.shape[1], nda.shape[2], 1))
    return cast(np.ndarray, nda_adjusted)


def shuffle(
    x_orig: np.ndarray, y_orig: np.ndarray, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle x and y in the same way."""
    np.random.seed(seed)
    idx = np.random.permutation(len(x_orig))
    return x_orig[idx], y_orig[idx]


def get_partition(
    x_orig: np.ndarray, y_orig: np.ndarray, partition: int, num_clients: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a single partition of an equally partitioned dataset."""
    step_size = len(x_orig) / num_clients
    start_index = int(step_size * partition)
    end_index = int(start_index + step_size)
    return x_orig[start_index:end_index], y_orig[start_index:end_index]

def get_partition_v2(
    x_orig: np.ndarray, y_orig: np.ndarray, client_id: int, num_sanples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a single partition of an equally partitioned dataset."""
    return x_orig[client_id*num_sanples:(client_id+1)*num_sanples], y_orig[client_id*num_sanples:(client_id+1)*num_sanples]
#    return x_orig, y_orig
