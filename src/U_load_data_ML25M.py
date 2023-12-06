import U_utils as utils
import U_data as data
import numpy as np
import pandas as pd


def load_data(data_path, trial, use_raw_cf=False):
    timer = utils.timer(name='main').tic()
    item_content_file = data_path + '/mv-tag-emb.npy'

    train_file = data_path + "/BPR_cv/BPR_tr_{trial}.tsv".format(trial=trial)
    test_file = data_path + "/BPR_cv/cold_movies_rating_test_{trial}.tsv".format(trial=trial)
    validation_file = data_path + "/BPR_cv/cold_movies_rating_vali_{trial}.tsv".format(trial=trial)
    user_cf_vec_file = data_path + "/BPR_cv/BPR_uvec_{trial}.npy".format(trial=trial)
    item_cf_vec_file = data_path + "/BPR_cv/BPR_ivec_{trial}.npy".format(trial=trial)

    info_file = data_path + "/stats.tsv"
    info = pd.read_csv(info_file, dtype=np.int32, delimiter='\t')
    num_users = info['users'][0]
    num_items = info['movies'][0]

    timer.toc('loaded num_users:%d, num_items:%d' % (num_users, num_items))

    dat = {}
    # load preference data
    timer.tic()
    user_cf_vec = np.load(user_cf_vec_file)
    item_cf_vec = np.load(item_cf_vec_file)
    dat['num_users'] = num_users
    dat['num_items'] = num_items
    dat['user_cf_vec'] = user_cf_vec
    dat['item_cf_vec'] = item_cf_vec

    if use_raw_cf:
        dat['user_cf_raw'] = user_cf_vec
        dat['item_cf_raw'] = item_cf_vec

    timer.toc('loaded U:%s,V:%s' % (str(user_cf_vec.shape), str(item_cf_vec.shape)))

    timer.tic()
    # pre-process
    _, dat['user_cf_vec'] = utils.standardize(dat['user_cf_vec'])
    _, dat['item_cf_vec'] = utils.standardize_2(dat['item_cf_vec'])
    timer.toc('standardized U,V')

    # load content data
    timer.tic()
    item_content_vec = np.load(item_content_file)
    _, item_content_vec = utils.standardize(item_content_vec)
    dat['item_content_vec'] = item_content_vec
    timer.toc('loaded item feature sparse matrix: %s' % (str(item_content_vec.shape)))

    # load split
    timer.tic()
    uid_col = "uid"
    iid_col = "mid"
    dtype = {uid_col: np.int32, iid_col: np.int32}
    train = pd.read_csv(train_file, sep='\t', usecols=[uid_col, iid_col], dtype=dtype)
    dat['user_list'] = train[uid_col].values
    dat['item_list'] = train[iid_col].values
    dat['user_indices'] = np.unique(train[uid_col].values)
    timer.toc('read train triplets %s' % str(train.shape)).tic()

    dat['test_eval'] = data.load_eval_data_ML(test_file=test_file, uid_col=uid_col, iid_col=iid_col)
    dat['validation_eval'] = data.load_eval_data_ML(test_file=validation_file, uid_col=uid_col, iid_col=iid_col)
    return dat

