import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN


class GAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        attns = []
        for _ in range(n_heads[0]):
            attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                                          out_sz=hid_units[0], activation=activation,
                                          in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
                                              out_sz=hid_units[i], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)

        out = []
        for i in range(n_heads[-1]):
            out.append(layers.attn_head(h_1, bias_mat=bias_mat,
                                        out_sz=nb_classes, activation=lambda x: x,
                                        in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]

        return logits

    def inference_AdjandFeature(inputs, adj, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                                bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        # structure att
        print(adj.shape)
        print(type(adj))
        print(type(bias_mat))
        bias_mat = bias_mat * 0.0
        stru_coef = layers.attn_coef(adj, bias_mat=bias_mat * 0.0,
                                     out_sz=nb_nodes, activation=activation,
                                     in_drop=ffd_drop, coef_drop=attn_drop, residual=False)
        margin = 0.005
        tadj = tf.convert_to_tensor(adj)
        adj = tf.add(tadj, stru_coef)
        print(adj.shape)
        one = tf.ones_like(bias_mat)
        zero = tf.zeros_like(bias_mat)
        bias_mat = tf.where(adj > margin, zero, one * -1e9)
        adj_refactor = tf.where(adj > margin, one, zero)
        attns = []
        for _ in range(n_heads[0]):
            attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                                          out_sz=hid_units[0], activation=activation,
                                          in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
                                              out_sz=hid_units[i], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)


        out = []
        for i in range(n_heads[-1]):
            out.append(layers.attn_head(h_1, bias_mat=bias_mat,
                                        out_sz=nb_classes, activation=lambda x: x,
                                        in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]

        # return logits, stru_coef
        return logits, adj_refactor

    # thresh version
    def inference_thresh(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                         bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False, thresh=0.005):
        print(inputs.shape)
        print('Thresh')
        print(thresh)
        attns = []
        for _ in range(n_heads[0]):
            attns.append(layers.attn_head_thresh(inputs, bias_mat=bias_mat,
                                                 out_sz=hid_units[0], activation=activation,
                                                 in_drop=ffd_drop, coef_drop=attn_drop, residual=False, thresh=thresh))
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.attn_head_thresh(h_1, bias_mat=bias_mat,
                                                     out_sz=hid_units[i], activation=activation,
                                                     in_drop=ffd_drop, coef_drop=attn_drop, residual=residual,
                                                     thresh=thresh))
            h_1 = tf.concat(attns, axis=-1)

        out = []
        for i in range(n_heads[-1]):
            out.append(layers.attn_head_thresh(h_1, bias_mat=bias_mat,
                                               out_sz=nb_classes, activation=lambda x: x,
                                               in_drop=ffd_drop, coef_drop=attn_drop, residual=False, thresh=thresh))
        logits = tf.add_n(out) / n_heads[-1]

        return logits

    def inference_sameatt(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                          bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        # print(inputs.shape)
        attns = []
        at_val = []
        for _ in range(n_heads[0]):
            at, at_val = layers.attn_head_same(inputs, bias_mat=bias_mat,
                                               out_sz=hid_units[0], activation=activation,
                                               in_drop=ffd_drop, coef_drop=attn_drop, residual=False)
            attns.append(at)
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.attn_head_same(h_1, bias_mat=bias_mat,
                                                   out_sz=hid_units[i], activation=activation,
                                                   in_drop=ffd_drop, coef_drop=attn_drop, residual=residual)[0])
            h_1 = tf.concat(attns, axis=-1)

        out = []
        for i in range(n_heads[-1]):
            out.append(layers.attn_head(h_1, bias_mat=bias_mat,
                                        out_sz=nb_classes, activation=lambda x: x,
                                        in_drop=ffd_drop, coef_drop=attn_drop, residual=False)[0])
        logits = tf.add_n(out) / n_heads[-1]

        return logits, at_val

    def inference_att(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                      bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        # print(inputs.shape)
        attns = []
        for _ in range(n_heads[0]):
            at, coefs = layers.attn_head_coef(inputs, bias_mat=bias_mat,
                                              out_sz=hid_units[0], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=False)
            attns.append(at)
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
                                              out_sz=hid_units[i], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)

        out = []
        for i in range(n_heads[-1]):
            out.append(layers.attn_head(h_1, bias_mat=bias_mat,
                                        out_sz=nb_classes, activation=lambda x: x,
                                        in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]

        return logits, coefs
