import tensorflow as tf
import tensorflow.keras as keras

from pathlib import Path
from collections import defaultdict
from itertools import cycle, permutations
from functools import partial

import numpy as np


class DataLoader:
    def __init__(self, topo_hash, node_hash, edge_hash):
        self.topo_hash = topo_hash
        self.node_hash = node_hash
        self.edge_hash = edge_hash

    def save_state(self, filename):
        np.savez(filename,
            topo_hash=self.topo_hash,
            node_hash=self.node_hash,
            edge_hash=self.edge_hash,
        )

    @staticmethod
    def from_state(filename):
        data = np.load(filename, allow_pickle=True)
        topologies = data["topo_hash"].item()
        node_hash = data["node_hash"].tolist()
        edge_hash = data["edge_hash"].tolist()

        return DataLoader(topologies, node_hash, edge_hash)

    def key2index(self, key):
        if not isinstance(key, str):
            key = key.numpy().decode("utf-8")

        tokens = key.split("+")

        nodes = [v for v in tokens if v.startswith("N")]
        edges = [v for v in tokens if v.startswith("E")]

        topo_index = [self.topo_hash[tokens[0]]]

        node_index = [-1] * 3
        for i, v in enumerate(nodes):
            node_index[i] = self.node_hash[v]

        edge_index = [-1] * 3
        for i, v in enumerate(edges):
            edge_index[i] = self.edge_hash[v]

        index = topo_index + node_index + edge_index

        return np.array(index)

    def make_dataset(self, keys, ys=None,
                     batch_size=32, buffer_size=10000,
                     repeat=True, shuffle=True):
        # Helper function.
        # DO NOT use Tout=[tf.int32] to prevent useless tuple output after map.
        # Just use Tout=tf.int32.
        def key2index(key):
            return tf.py_function(self.key2index, inp=[key], Tout=tf.int32)

        xs = tf.data.Dataset.from_generator(
            lambda: (key for key in keys),
            output_types=tf.string,
        )

        xs = xs.map(key2index)
        # Cache after map to improve the performance.
        xs = xs.cache()

        if ys is not None:
            if len(ys.shape) == 1:
                ys = np.reshape(ys, (-1, 1))

            _ys = ys
            ys = tf.data.Dataset.from_generator(
                lambda: (y for y in _ys),
                output_types=tf.float32,
            )
            dataset = tf.data.Dataset.zip((xs, ys))
        else:
            dataset = xs

        if repeat:
            dataset = dataset.repeat()
        if shuffle:
            dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size)

        return dataset


class SelfWeight(keras.layers.Layer):
    def __init__(self, emb_size):
        super().__init__()

        self.emb_size = emb_size
        units = 3*emb_size*emb_size
        self.dense = keras.layers.Dense(units=units, activation=tf.math.tanh)

    def call(self, x):
        B = x.shape[0]

        x = self.dense(x)
        weight = tf.reshape(x, [B, 3, self.emb_size, self.emb_size])

        return weight


class InteractionWeight(keras.layers.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size

        units = in_size * out_size
        self.dense = keras.layers.Dense(units=units, activation=tf.math.tanh)

    def call(self, x):
        B = x.shape[0]

        x = self.dense(x)
        weight = tf.reshape(x, [B, self.in_size, self.out_size])

        return weight

# Naming convention: When using CamelCase names, capitalize all letters of an
#                    abbreviation (e.g. HTTPServer)
class MOFNet(keras.Model):
    def __init__(self):
        super().__init__()

        topo_emb_size = 128
        node_emb_size = 128
        edge_emb_size = 128

        self.topo_embedding = keras.layers.Embedding(
                                  input_dim=3000, output_dim=topo_emb_size)
        self.node_embedding = keras.layers.Embedding(
                                  input_dim=1000, output_dim=node_emb_size)
        self.edge_embedding = keras.layers.Embedding(
                                  input_dim=300, output_dim=edge_emb_size)

        self.dropout = keras.layers.Dropout(0.5)

        self.node_weight = SelfWeight(node_emb_size)
        self.edge_weight = SelfWeight(edge_emb_size)

        self.node_batchnorm = keras.layers.BatchNormalization()
        self.edge_batchnorm = keras.layers.BatchNormalization()

        first_size = 3*node_emb_size + 3*edge_emb_size
        self.interaction_weight = InteractionWeight(first_size, 64)
        self.interaction_batchrnom = keras.layers.BatchNormalization()

        self.hidden_dense = keras.layers.Dense(units=32, use_bias=False)
        self.hidden_batchnorm = keras.layers.BatchNormalization()

        self.output_dense = keras.layers.Dense(units=1)

    def initialize_weights(self):
        self(np.zeros([1, 7], dtype=np.float32))

    @tf.function
    def call(self, x, training=False):
        # B: batch size.
        B = x.shape[0]

        # Split input vector to topology, nodes and edges.
        topo_x, node_x, edge_x = tf.split(x, [1, 3, 3], axis=1)

        # Make topology embedding. . ---------------------------------
        topo_emb = self.topo_embedding(topo_x)
        topo_emb = tf.reshape(topo_emb, [B, -1])
        # Dropout to topology embedding.
        topo_emb = self.dropout(topo_emb, training=training)

        # Make node embedding. ---------------------------------------
        node_x = tf.where(node_x >= 0, node_x, tf.zeros_like(node_x))
        node_emb = self.node_embedding(node_x)
        node_emb = self.dropout(node_emb, training=training)

        # Apply self interaction in topology.
        # node_weight: (B, 3, E, E), node_emb: (B, 3, E).
        node_weight = self.node_weight(topo_emb)
        node_emb = tf.einsum("ijkl,ijl->ijk", node_weight, node_emb)
        node_emb = self.node_batchnorm(node_emb, training=training)
        node_emb = tf.nn.relu(node_emb)
        # Apply mask and reshape to [B, 3*node_emb_size]
        shape = node_x.shape + [self.node_embedding.output_dim]
        mask = tf.broadcast_to(node_x[:, :, tf.newaxis], shape=shape)
        mask = (mask >= 0)
        node_emb = tf.where(mask, node_emb, tf.zeros_like(node_emb))
        node_emb = tf.reshape(node_emb, [B, -1])

        # Make edge embedding. ---------------------------------------
        edge_x = tf.where(edge_x >= 0, edge_x, tf.zeros_like(edge_x))
        edge_emb = self.edge_embedding(edge_x)
        edge_emb = self.dropout(edge_emb, training=training)

        # Apply self interaction in topology.
        # edge_weight: (B, 3, E, E), edge_emb: (B, 3, E).
        edge_weight = self.edge_weight(topo_emb)
        edge_emb = tf.einsum("ijkl,ijl->ijk", edge_weight, edge_emb)
        edge_emb = self.edge_batchnorm(edge_emb, training=training)
        edge_emb = tf.nn.relu(edge_emb)
        # Apply mask and reshape to [B, 3*node_emb_size]
        shape = edge_x.shape + [self.edge_embedding.output_dim]
        mask = tf.broadcast_to(edge_x[:, :, tf.newaxis], shape=shape)
        mask = (mask >= 0)
        edge_emb = tf.where(mask, edge_emb, tf.zeros_like(edge_emb))
        edge_emb = tf.reshape(edge_emb, [B, -1])

        # Concatenate building block tensors.
        x = tf.concat([node_emb, edge_emb], axis=1)

        # Apply interaction.
        weight = self.interaction_weight(topo_emb)
        x = tf.einsum("ijk,ij->ik", weight, x)
        x = self.interaction_batchrnom(x, training=training)
        x = tf.nn.relu(x)

        x = self.hidden_dense(x)
        x = self.hidden_batchnorm(x, training=training)
        x = tf.nn.relu(x)

        x = self.output_dense(x)

        return x
