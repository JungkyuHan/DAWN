import U_utils
import pickle
import U_data as data
import numpy as np
from sklearn import datasets
import pandas as pd
import scipy.sparse as sp


def load_data(data_name, use_raw_cf=False):
    timer = U_utils.timer(name='main').tic()
    data_path = '../data/' + data_name
    item_content_file = data_path + '/item_features.txt'
    train_file = data_path + '/train.csv'
    test_file = data_path + '/test.csv'
    validation_file = data_path + '/vali.csv'
    user_cf_vec_file = data_path + '/U_BPR.npy'
    item_cf_vec_file = data_path + '/V_BPR.npy'
    with open(data_path + '/info.pkl', 'rb') as f:
        info = pickle.load(f)
        num_user = info['num_user']
        num_item = info['num_item']
    timer.toc('loaded num_users:%d, num_items:%d' % (num_user, num_item))

    dat = {}
    dat['num_users'] = num_user
    dat['num_items'] = num_item

    # load preference data
    timer.tic()
    user_cf_vec = np.load(user_cf_vec_file)
    item_cf_vec = np.load(item_cf_vec_file)

    dat['user_cf_vec'] = user_cf_vec
    dat['item_cf_vec'] = item_cf_vec

    if use_raw_cf:
        dat['user_cf_raw'] = user_cf_vec
        dat['item_cf_raw'] = item_cf_vec

    timer.toc('loaded U:%s,V:%s' % (str(user_cf_vec.shape), str(item_cf_vec.shape)))

    timer.tic()
    # pre-process
    _, dat['user_cf_vec'] = U_utils.standardize(dat['user_cf_vec'])
    _, dat['item_cf_vec'] = U_utils.standardize_2(dat['item_cf_vec'])
    timer.toc('standardized U,V')

    # load content data
    timer.tic()
    item_content_vec, _ = datasets.load_svmlight_file(item_content_file, zero_based=True, dtype=np.float32)
    item_content_vec = tfidf(item_content_vec)
    from sklearn.utils.extmath import randomized_svd
    u, s, _ = randomized_svd(item_content_vec, n_components=300, n_iter=5)
    item_content_vec = u * s

    _, item_content_vec = U_utils.standardize(item_content_vec)
    dat['item_content_vec'] = item_content_vec
    timer.toc('loaded item feature sparse matrix: %s' % (str(item_content_vec.shape)))

    # load split
    timer.tic()
    train = pd.read_csv(train_file, dtype=np.int32)
    dat['user_list'] = train['uid'].values
    dat['item_list'] = train['iid'].values
    dat['user_indices'] = np.unique(train['uid'].values)
    timer.toc('read train triplets %s' % str(train.shape)).tic()

    dat['test_eval'] = data.load_eval_data(test_file)
    dat['validation_eval'] = data.load_eval_data(validation_file)
    return dat


def tfidf(R):
    row = R.shape[0]
    col = R.shape[1]
    Rbin = R.copy()
    Rbin[Rbin != 0] = 1.0
    R = R + Rbin
    tf = R.copy()
    tf.data = np.log(tf.data)
    idf = np.sum(Rbin, 0)
    idf = np.log(row / (1 + idf))
    idf = sp.spdiags(idf, 0, col, col)
    return tf * idf
