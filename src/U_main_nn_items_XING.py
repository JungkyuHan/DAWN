import tensorflow as tf
import argparse
import U_contents_sim as contents_sim
import numpy as np
import U_load_data_XING as load_data_XING


def main():
    print("///**** Main_NN_Items XING ****///")
    data_name = args.data
    dat = load_data_XING.load_data(data_name)

    item_list = dat['item_list']
    item_warm = np.unique(item_list)

    item_content = dat['item_content_vec']
    item_content = item_content.toarray()

    item_all = np.arange(0, item_content.shape[0])
    item_cold = np.setdiff1d(item_all, item_warm)

    warm_item_content = np.copy(item_content)
    warm_item_lengths = np.linalg.norm(warm_item_content, axis=-1)
    zero_content = np.zeros([1, item_content.shape[1]])
    warm_item_content[item_cold, :] = zero_content
    warm_item_content[warm_item_lengths > 0] = warm_item_content[warm_item_lengths > 0] \
                                               / warm_item_lengths[warm_item_lengths > 0][:, np.newaxis]
    warm_item_content_tensor_normalized = tf.convert_to_tensor(value=warm_item_content, dtype=tf.float32)

    item_content_tensor = tf.convert_to_tensor(value=item_content, dtype=tf.float32)
    item_content_tensor_normalized, _ = tf.linalg.normalize(item_content_tensor, ord='euclidean', axis=1)

    sim_mat = contents_sim.get_item_sim_mat(item_by_ctx_mat=item_content_tensor_normalized,
                                           warm_item_by_ctx_mat=warm_item_content_tensor_normalized)

    top_k_vals, top_k_indices = contents_sim.get_top_k_nn(sim_mat, top_k=10)
    top_k_indices_ndarray = top_k_indices.numpy()
    top_k_vals_ndarray = top_k_vals.numpy()
    np.save('../data/XING/top_k_nn_item_index.npy', top_k_indices_ndarray)
    np.save('../data/XING/top_k_nn_item_sim.npy', top_k_vals_ndarray)
    print("///**** Calculation Finished ****///")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main nn items XING",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str, default='XING', help='path to eval in the downloaded folder')
    args = parser.parse_args()

    main()
