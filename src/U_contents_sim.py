import tensorflow as tf
import numpy as np


# Assume each row of item_by_ctx_mat is normalized vector (=size 1)
# item_by_ctx_mat : the number of items * dimension of context vectors
# cold_start_items : Index of items whose CF vectors are not available.
def get_item_sim_mat(item_by_ctx_mat, warm_item_by_ctx_mat=None, zero_diagonal=True):
    if warm_item_by_ctx_mat is None:
        sim_mat = tf.linalg.matmul(a=item_by_ctx_mat, b=item_by_ctx_mat, transpose_b=True)
    else:
        sim_mat = tf.linalg.matmul(a=item_by_ctx_mat, b=warm_item_by_ctx_mat, transpose_b=True)

    if zero_diagonal:
        dim = tf.shape(sim_mat)[0]
        diagonal = np.zeros(shape=[dim])
        sim_mat = tf.linalg.set_diag(input=sim_mat, diagonal=diagonal)

    return sim_mat


def get_top_k_nn(sim_mat, top_k):
    (top_k_values, top_k_indices) = tf.math.top_k(sim_mat, k=top_k, sorted=True)
    return top_k_values, top_k_indices

