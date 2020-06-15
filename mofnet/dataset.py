import numpy as np

import tensorflow as tf
import tensorflow.keras as keras


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
