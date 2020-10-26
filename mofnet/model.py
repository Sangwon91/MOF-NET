import tensorflow as tf
import tensorflow.keras as keras


class SelfWeight(keras.layers.Layer):
    def __init__(self, emb_size, out_size=None):
        super().__init__()

        if out_size is None:
            self.out_size = emb_size
        else:
            self.out_size = out_size

        self.emb_size = emb_size

        units = 3 * self.out_size * self.emb_size
        self.dense = keras.layers.Dense(units=units, activation=tf.math.tanh)

    def call(self, x):
        B = x.shape[0]

        x = self.dense(x)
        weight = tf.reshape(x, [B, 3, self.out_size, self.emb_size])

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
    def __init__(self, **kwargs):
        super().__init__()

        default_parameters = {
            "num_topos": 3000,
            "num_nodes": 1000,
            "num_edges": 300,
            "topo_emb_size": 128,
            "node_emb_size": 128,
            "edge_emb_size": 128,
            "node_self_size": 128,
            "edge_self_size": 128,
            "mof_emb_size": 64,
            "dropout_rate": 0.5,
            "activation": tf.nn.relu,
        }

        params = {}
        for key in default_parameters.keys():
            if key in kwargs:
                params[key] = kwargs[key]
            else:
                params[key] = default_parameters[key]

        num_topos = params["num_topos"]
        num_nodes = params["num_nodes"]
        num_edges = params["num_edges"]
        topo_emb_size = params["topo_emb_size"]
        node_emb_size = params["node_emb_size"]
        edge_emb_size = params["edge_emb_size"]
        node_self_size = params["node_self_size"]
        edge_self_size = params["edge_self_size"]
        mof_emb_size = params["mof_emb_size"]
        dropout_rate = params["dropout_rate"]
        self.activation = params["activation"]

        self.topo_embedding = keras.layers.Embedding(
                                  input_dim=num_topos,
                                  output_dim=topo_emb_size,
                              )
        self.node_embedding = keras.layers.Embedding(
                                  input_dim=num_nodes,
                                  output_dim=node_emb_size,
                              )
        self.edge_embedding = keras.layers.Embedding(
                                  input_dim=num_edges,
                                  output_dim=edge_emb_size,
                              )

        self.dropout = keras.layers.Dropout(dropout_rate)

        self.node_weight = SelfWeight(node_emb_size, node_self_size)
        self.edge_weight = SelfWeight(edge_emb_size, edge_self_size)

        self.node_batchnorm = keras.layers.BatchNormalization()
        self.edge_batchnorm = keras.layers.BatchNormalization()

        first_size = 3*node_self_size + 3*edge_self_size
        self.interaction_weight = InteractionWeight(first_size, mof_emb_size)
        self.interaction_batchrnom = keras.layers.BatchNormalization()

        self.hidden_dense = keras.layers.Dense(units=32, use_bias=False)
        self.hidden_batchnorm = keras.layers.BatchNormalization()

        self.output_dense = keras.layers.Dense(units=1)

    def initialize_weights(self):
        self.call(tf.zeros(shape=[1, 7]))
        self.built = True

    @tf.function
    def calculate_mof_embedding(self, x, training=False, dropout=None):
        if dropout is None:
            dropout = training

        activation = self.activation
        # B: batch size.
        B = x.shape[0]

        # Split input vector to topology, nodes and edges.
        topo_x, node_x, edge_x = tf.split(x, [1, 3, 3], axis=1)

        # Make topology embedding. . ---------------------------------
        topo_emb = self.topo_embedding(topo_x)
        topo_emb = tf.reshape(topo_emb, [B, -1])
        # Dropout to topology embedding.
        topo_emb = self.dropout(topo_emb, training=dropout)

        # Make node embedding. ---------------------------------------
        node_x = tf.where(node_x >= 0, node_x, tf.zeros_like(node_x))
        node_emb = self.node_embedding(node_x)
        node_emb = self.dropout(node_emb, training=dropout)

        # Apply self interaction in topology.
        # node_weight: (B, 3, Eout, E), node_emb: (B, 3, E).
        # Result: (B, 3, Eout).
        node_weight = self.node_weight(topo_emb)
        node_emb = tf.einsum("ijkl,ijl->ijk", node_weight, node_emb)
        node_emb = self.node_batchnorm(node_emb, training=training)
        node_emb = activation(node_emb)
        # Apply mask and reshape to [B, 3*node_self_size]
        shape = node_x.shape + [self.node_weight.out_size]
        mask = tf.broadcast_to(node_x[:, :, tf.newaxis], shape=shape)
        mask = (mask >= 0)
        node_emb = tf.where(mask, node_emb, tf.zeros_like(node_emb))
        node_emb = tf.reshape(node_emb, [B, -1])

        # Make edge embedding. ---------------------------------------
        edge_x = tf.where(edge_x >= 0, edge_x, tf.zeros_like(edge_x))
        edge_emb = self.edge_embedding(edge_x)
        edge_emb = self.dropout(edge_emb, training=dropout)

        # Apply self interaction in topology.
        # edge_weight: (B, 3, Eout, E), edge_emb: (B, 3, E).
        # Result: (B, 3, Eout).
        edge_weight = self.edge_weight(topo_emb)
        edge_emb = tf.einsum("ijkl,ijl->ijk", edge_weight, edge_emb)
        edge_emb = self.edge_batchnorm(edge_emb, training=training)
        edge_emb = activation(edge_emb)
        # Apply mask and reshape to [B, 3*node_self_size]
        shape = edge_x.shape + [self.edge_weight.out_size]
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
        x = activation(x)

        return x

    @tf.function
    def call(self, x, training=False, dropout=None):
        # x: (B, 64).
        x = self.calculate_mof_embedding(x, training, dropout)

        # Apply MLP to predict property.
        # x: (B, 32).
        x = self.hidden_dense(x)
        x = self.hidden_batchnorm(x, training=training)
        x = self.activation(x)

        # x: (B, 1).
        x = self.output_dense(x)

        return x
