import tensorflow as tf
import numpy as np
from tqdm import tqdm
import scipy


class EvalMetricCalculator:
    def __init__(self, top_k_list=[10]):
        self.num_calculate_metric_called = 0
        self.num_recs = 0

        self.predictions = []
        self.recall_sum_list = []
        self.precision_sum_list = []
        self.num_hits_list = []
        self.num_gts_list = []

        self.top_k_list = top_k_list
        self.max_k = 0

        for k in self.top_k_list:
            if self.max_k < k:
                self.max_k = k

        list_len = len(top_k_list)
        for i in range(list_len):
            self.recall_sum_list.append(0.0)
            self.precision_sum_list.append(0.0)
            self.num_hits_list.append(0)
            self.num_gts_list.append(0)

        ## IDCG Calculation
        self.idcg_array = np.arange(self.max_k) + 1
        self.idcg_array = 1 / np.log2(self.idcg_array + 1)
        self.idcg_table = np.zeros(self.max_k)
        for i in range(self.max_k):
            self.idcg_table[i] = np.sum(self.idcg_array[:(i + 1)])

    @staticmethod
    def convert_top_k_idx_list_2_sparse_mat(top_k_index_list, num_items):
        idx_array = []
        num_users = len(top_k_index_list)
        for i, top_k_list in enumerate(top_k_index_list):
            for j in top_k_list:
                idx_array.append((i, j))
        len_idx_array = len(idx_array)
        sparse_tensor = tf.sparse.SparseTensor(
            indices=idx_array,
            values=np.full(len_idx_array, 1.0, dtype=np.float32),
            dense_shape=[num_users, num_items])
        return sparse_tensor

    def collect_predictions(self, top_k_index_mat):
        self.predictions.append(top_k_index_mat)

    def get_metric(self, test_data):
        tf_eval_preds = np.concatenate(self.predictions)

        # filter non-zero targets
        y_nz = [len(x) > 0 for x in test_data.R_test_inf.rows]
        y_nz = np.arange(len(test_data.R_test_inf.rows))[y_nz]

        preds_all = tf_eval_preds[y_nz, :]

        recall = []
        precision = []
        ndcg = []
        num_hits = []
        num_gts = []

        for at_k in self.top_k_list:
            preds_k = preds_all[:, :at_k]
            y = test_data.R_test_inf[y_nz, :]

            num_x_vals = preds_k.shape[0] * preds_k.shape[1]
            r_idxes = []
            c_idxes = []
            for r_idx, row_arr in enumerate(preds_k):
                for c_idx in row_arr:
                    r_idxes.append(r_idx)
                    c_idxes.append(c_idx)

            x = scipy.sparse.coo_matrix(
                (np.ones(num_x_vals), (r_idxes, c_idxes)),
                shape=y.shape
            ).tolil(copy=False)

            z = y.multiply(x)

            num_hits_k = np.sum(z, 1)
            num_gts_k = np.sum(y, 1)
            recall.append(np.mean(np.divide(num_hits_k, num_gts_k)))
            precision.append(np.mean(num_hits_k / at_k))

            num_hits_total = np.sum(num_hits_k)
            num_gts_total = np.sum(num_gts_k)
            num_hits.append(num_hits_total)
            num_gts.append(num_gts_total)

            x_coo = x.tocoo()
            rows = x_coo.row
            cols = x_coo.col
            y_csr = y.tocsr()
            dcg_array = y_csr[(rows, cols)].A1.reshape((preds_k.shape[0], -1))
            dcg = np.sum(dcg_array * self.idcg_array[:at_k].reshape((1, -1)), axis=1)
            idcg = np.sum(y, axis=1) - 1
            idcg[np.where(idcg >= at_k)] = at_k - 1
            idcg = self.idcg_table[idcg.astype(int)]
            ndcg.append(np.mean(dcg / idcg))

        return precision, recall, ndcg, num_hits, num_gts

    def get_batch_weights(self):
        batch_weights = []
        weight_sum = 0.0
        for tf_eval_preds in self.predictions:
            batch_size = float(tf_eval_preds.shape[0])
            batch_weights.append(batch_size)
            weight_sum += batch_size
        for i, val in enumerate(batch_weights):
            batch_weights[i] = batch_weights[i] / weight_sum
        return batch_weights

    @staticmethod
    def metric_sum(metric_batch_list):
        metric_sum = []

        for _ in metric_batch_list[0]:
            metric_sum.append(0)

        for metric in metric_batch_list:
            for j, val in enumerate(metric):
                metric_sum[j] += val

        return metric_sum

    @staticmethod
    def metric_weighted_avg(metric_batch_list, batch_weights):
        metric_w_avg = []

        for _ in metric_batch_list[0]:
            metric_w_avg.append(0.0)

        for i, metric in enumerate(metric_batch_list):
            for j, val in enumerate(metric):
                metric_w_avg[j] += batch_weights[i] * val

        return metric_w_avg

    def get_metric_distributed(self, test_data):
        start_batch_idx_list = []
        end_batch_idx_list = []
        for tf_eval_preds in self.predictions:
            pred_len = tf_eval_preds.shape[0]
            end_batch_idx_list.append(pred_len)
        num_batchs = len(end_batch_idx_list)

        start_batch_idx_list.append(0)
        for i in range(1, num_batchs):
            end_batch_idx_list[i] = end_batch_idx_list[i] + end_batch_idx_list[i - 1]
            start_batch_idx_list.append(end_batch_idx_list[i - 1])

        recall_batch = []
        precision_batch = []
        ndcg_batch = []
        num_hits_batch = []
        num_gts_batch = []

        for i, tf_eval_preds in enumerate(self.predictions):
            eval_preds = tf_eval_preds.numpy()
            start_batch_idx = start_batch_idx_list[i]
            end_batch_idx = end_batch_idx_list[i]
            R_test_inf_rows = test_data.R_test_inf.rows[start_batch_idx:end_batch_idx]

            y_nz = [len(x) > 0 for x in R_test_inf_rows]
            y_nz = np.arange(len(R_test_inf_rows))[y_nz]

            # filter non-zero targets
            preds_all = eval_preds[y_nz, :]

            recall = []
            precision = []
            ndcg = []
            num_hits = []
            num_gts = []

            for at_k in self.top_k_list:
                preds_k = preds_all[:, :at_k]
                y = test_data.R_test_inf[start_batch_idx:end_batch_idx, :]

                num_x_vals = preds_k.shape[0] * preds_k.shape[1]
                r_idxes = []
                c_idxes = []
                for r_idx, row_arr in enumerate(preds_k):
                    for c_idx in row_arr:
                        r_idxes.append(r_idx)
                        c_idxes.append(c_idx)

                x = scipy.sparse.coo_matrix(
                    (np.ones(num_x_vals), (r_idxes, c_idxes)),
                    shape=y.shape
                ).tolil(copy=False)

                z = y.multiply(x)

                num_hits_k = np.sum(z, 1)
                num_gts_k = np.sum(y, 1)
                recall.append(np.mean(np.divide(num_hits_k, num_gts_k)))
                precision.append(np.mean(num_hits_k / at_k))

                num_hits_total = np.sum(num_hits_k)
                num_gts_total = np.sum(num_gts_k)
                num_hits.append(num_hits_total)
                num_gts.append(num_gts_total)

                x_coo = x.tocoo()
                rows = x_coo.row
                cols = x_coo.col
                y_csr = y.tocsr()
                dcg_array = y_csr[(rows, cols)].A1.reshape((preds_k.shape[0], -1))
                dcg = np.sum(dcg_array * self.idcg_array[:at_k].reshape((1, -1)), axis=1)
                idcg = np.sum(y, axis=1) - 1
                idcg[np.where(idcg >= at_k)] = at_k - 1
                idcg = self.idcg_table[idcg.astype(int)]
                ndcg.append(np.mean(dcg / idcg))

            recall_batch.append(recall)
            precision_batch.append(precision)
            ndcg_batch.append(ndcg)
            num_hits_batch.append(num_hits)
            num_gts_batch.append(num_gts)

        b_weights = self.get_batch_weights()

        f_recall = EvalMetricCalculator.metric_weighted_avg(recall_batch, b_weights)
        f_prec = EvalMetricCalculator.metric_weighted_avg(precision_batch, b_weights)
        f_ndcg = EvalMetricCalculator.metric_weighted_avg(ndcg_batch, b_weights)
        f_num_hits = EvalMetricCalculator.metric_sum(num_hits_batch)
        f_num_gts = EvalMetricCalculator.metric_sum(num_gts_batch)

        return f_prec, f_recall, f_ndcg, f_num_hits, f_num_gts

    @staticmethod
    def print_test_result(top_k_list, precision_list, recall_list, ndcg_list, num_hits_list, num_gts_list,
                          result_label="Test"):
        print("[%s result]" % result_label)

        print("k\tPrec\tRec\tNdcg\tHit\tGt")
        for i, k in enumerate(top_k_list):
            print("%d\t%.3f\t%.3f\t%.3f\t%d\t%d"
                  % (k, precision_list[i], recall_list[i], ndcg_list[i], num_hits_list[i], num_gts_list[i]))

