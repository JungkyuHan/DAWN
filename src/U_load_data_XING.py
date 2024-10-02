import U_utils as utils
import pickle
import U_data as data
import numpy as np
import pandas as pd
import scipy.sparse as sp


def load_data(data_name, use_raw_cf=False):
    timer = utils.timer(name='main').tic()
    data_path = '../data/' + data_name
    user_cf_vec_file = data_path + '/U_BPR.npy'
    item_cf_vec_file = data_path + '/V_BPR.npy'
    user_content_file = data_path + '/user_content.npz'
    item_content_file = data_path + '/item_content.npz'
    train_file = data_path + '/train.csv'
    vali_item_file = data_path + '/vali_item.csv'
    test_item_file = data_path + '/test_item.csv'
    with open(data_path + '/info.pkl', 'rb') as f:
        info = pickle.load(f)
        num_user = info['num_user']
        num_item = info['num_item']

    dat = {}
    dat['num_users'] = num_user
    dat['num_items'] = num_item

    # load preference data
    timer.tic()
    user_cf_vec = np.load(user_cf_vec_file)
    item_cf_vec = np.load(item_cf_vec_file)

    dat['user_cf_vec'] = user_cf_vec
    dat['item_cf_vec'] = item_cf_vec
    timer.toc('loaded U:%s,V:%s' % (str(user_cf_vec.shape), str(item_cf_vec.shape)))

    if use_raw_cf:
        dat['user_cf_raw'] = user_cf_vec
        dat['item_cf_raw'] = item_cf_vec

    # pre-process
    _, dat['user_cf_vec'] = utils.standardize(dat['user_cf_vec'])
    _, dat['item_cf_vec'] = utils.standardize_3(dat['item_cf_vec'])
    timer.toc('standardized U,V')

    # load content data
    timer.tic()
    # user_content_vec = sp.load_npz(user_content_file)
    # dat['user_content_vec'] = user_content_vec.tolil(copy=False)
    item_content_vec = sp.load_npz(item_content_file)
    dat['item_content_vec'] = item_content_vec.tolil(copy=False)
    timer.toc('loaded item feature sparse matrix: %s' % (str(item_content_vec.shape))).tic()

    # load split
    timer.tic()
    train = pd.read_csv(train_file, dtype=np.int32)
    dat['user_list'] = train['uid'].values
    dat['item_list'] = train['iid'].values
    dat['user_indices'] = np.unique(train['uid'].values)
    dat['warm_item'] = np.unique(train['iid'].values)
    timer.toc('read train triplets %s' % str(train.shape)).tic()

    dat['test_item_eval'] = data.load_eval_data(test_item_file)
    # dat['test_user_eval'] = data.load_eval_data(test_user_file, cold_user=True, test_item_ids=dat['warm_item'])
    # dat['test_user_item_eval'] = data.load_eval_data(test_user_item_file)
    dat['vali_item_eval'] = data.load_eval_data(vali_item_file)
    # dat['vali_user_eval'] = data.load_eval_data(vali_user_file, cold_user=True, test_item_ids=dat['warm_item'])
    # dat['vali_user_item_eval'] = data.load_eval_data(vali_user_item_file)
    return dat
