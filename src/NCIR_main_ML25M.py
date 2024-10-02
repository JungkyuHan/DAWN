import numpy as np
import tensorflow as tf
from NCIR import NCIR
import U_load_data_ML25M as load_data_ML25M
import U_utils as utils

import argparse
import time


def print_datastat(trial, data_path):
    dat = load_data_ML25M.load_data(data_path=data_path, trial=trial)

    test_eval = dat['test_eval']
    validation_eval = dat['validation_eval']

    user_list = dat['user_list']
    item_list = dat['item_list']
    user_warm = np.unique(user_list)
    item_warm = np.unique(item_list)

    ##### For stat calulation -- start
    num_clicks = len(user_list)
    num_users = len(user_warm)
    num_items = len(item_warm)
    print("trial: %d" % trial)
    print("data\tnum_users\tnum_items\tnum_clicks")
    print("tr\t%d\t%d\t%d" % (num_users, num_items, num_clicks))
    num_users, num_items, num_clicks = validation_eval.get_stats()
    print("va\t%d\t%d\t%d" % (num_users, num_items, num_clicks))
    num_users, num_items, num_clicks = test_eval.get_stats()
    print("te\t%d\t%d\t%d" % (num_users, num_items, num_clicks))


def main(trial, data_path, dest_item_idx_file_name):
    seed = args.rand_seed
    if seed == 0 or seed > 1000:
        seed = seed
    else:
        seed = round(time.time() * 1000) % 1000000
    print("seed: %d" % seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    idx_gpu_used = args.gpu_idx
    list_gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in list_gpu:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_visible_devices(list_gpu[idx_gpu_used], 'GPU')

    item_idx_file_path = data_path + dest_item_idx_file_name.format(trial=trial)

    # manual param override - begin
    eval_top_k_list = [10, 20, 50, 100]
    num_epochs = 10  # 10, 20
    neg = 5

    cf_vec_dropout = None
    carry_out_test = True
    num_neighbors = 5

    lr = args.lr
    decay_lr_every = args.decay_lr_every  # learning rate decay for each epoch?
    lr_decay = args.lr_decay  # learning rate decay

    cf_pred_loss_w = 0.0000005
    layer_weight_reg = None
    num_experts = 5
    attention_vec_dim = 128

    if neg is None:
        neg = args.neg
    if cf_pred_loss_w is None:
        cf_pred_loss_w = args.cf_pred_loss_w
    if layer_weight_reg is None:
        layer_weight_reg = args.layer_weight_reg
    if num_neighbors is None:
        num_neighbors = args.num_neighbors
    if cf_vec_dropout is None:
        cf_vec_dropout = args.dropout

    # manual param override - end
    # params for data input
    data_batch_size = 1024 * 10

    # model layer description
    model_select = args.model_select  # hidden layer description
    # evaluation description
    eval_batch_size = 30000  # the batch size when test

    dat = load_data_ML25M.load_data(data_path=data_path, trial=trial)

    user_cf_vec = dat['user_cf_vec'] # normalized bpr user vec
    item_cf_vec = dat['item_cf_vec'] # normalized bpr item vec
    item_content_vec = dat['item_content_vec']

    test_eval = dat['test_eval']
    validation_eval = dat['validation_eval']

    user_list = dat['user_list']
    item_list = dat['item_list']
    item_warm = np.unique(item_list)
    v_neighbor_mat = np.load(item_idx_file_path)

    timer = utils.timer(name='main').tic()
    # prep eval
    timer.tic()
    test_eval.init_tf2(user_cf_vec, item_cf_vec, None, item_content_vec, None, v_neighbor_mat,
                       eval_batch_size, cold_user=False, cold_item=True)  # init data for evaluation
    validation_eval.init_tf2(user_cf_vec, item_cf_vec, None, item_content_vec, None, v_neighbor_mat,
                             eval_batch_size, cold_user=False, cold_item=True)  # init data for evaluation

    timer.toc('initialized eval data').tic()

    cf_vec_rank = user_cf_vec.shape[1]
    if attention_vec_dim is None:
        attention_vec_dim = cf_vec_rank

    item_content_vec_rank = item_content_vec.shape[1]

    csr_dawn = NCIR(cf_vec_rank=cf_vec_rank,
                        item_content_vec_rank=item_content_vec_rank,
                        attention_vec_dim=attention_vec_dim,
                        model_select=model_select, num_neighbors=num_neighbors, num_experts=num_experts,
                        learning_rate=lr, lr_decay=lr_decay, lr_decay_step=decay_lr_every,
                        cf_vec_dropout=cf_vec_dropout,
                        cf_vec_pred_loss_w=cf_pred_loss_w, layer_weight_reg=layer_weight_reg,
                        data_set_name="ML25M", model_dir="../model", trials=trial, seed=seed)
    csr_dawn.build_model()

    if not carry_out_test:
        test_eval = None

    csr_dawn.fit(user_tr_list=user_list, item_tr_list=item_list,
                 u_pref=user_cf_vec,
                 v_pref=item_cf_vec, v_content=item_content_vec, item_warm=item_warm, v_neighbor_mat=v_neighbor_mat,
                 num_negative_samples=neg, data_batch_size=data_batch_size, dropout=cf_vec_dropout,
                 epochs=num_epochs,
                 tuning_data=validation_eval, test_data=test_eval, eval_top_k_list=eval_top_k_list)
    print("Train finished. Bye~")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="main_ML25M",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-select', nargs='+', type=int,
                        default=[128],
                        help='specify the fully-connected architecture, starting from input,'
                             ' numbers indicate numbers of hidden units',
                        )
    parser.add_argument('--rank', type=int, default=200, help='output rank of latent model')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--eval-every', type=int, default=1, help='evaluate every X user-batch')
    parser.add_argument('--neg', type=float, default=5, help='negative sampling rate')
    parser.add_argument('--lr', type=float, default=0.005, help='starting learning rate')  # default = 0.005
    parser.add_argument('--lr_decay', type=float, default=0.8, help='learning rate decay')
    parser.add_argument('--decay_lr_every', type=int, default=10, help='decay step') # default : 10
    parser.add_argument('--cf_pred_loss_w', type=float, default=0.0001, help='cf_pred_loss parameter')
    parser.add_argument('--layer_weight_reg', type=float, default=0.0001, help='layer_regularization') # default 0.0001
    parser.add_argument('--num_neighbors', type=int, default=5, help='number of item neighbors')
    parser.add_argument('--gpu_idx', type=int, default=0, help='Using GPU Idx')
    parser.add_argument('--rand_seed', type=int, default=0, help='use random seed')

    args = parser.parse_args()
    args, _ = parser.parse_known_args()
    for key in vars(args):
        print(key + ":" + str(vars(args)[key]))

    data_path = "../data/ML25M"
    dest_item_idx_file_name = "/BPR_cv/top_k_nn_item_index_{trial}.npy"
    trial = 0
    main(trial=trial, data_path=data_path, dest_item_idx_file_name=dest_item_idx_file_name)
