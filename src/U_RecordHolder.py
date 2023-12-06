import numpy as np


class RecordHolder:
    def __init__(self, top_k_list=[10]):
        self.epoch = 0
        self.n_step = -1
        self.recall_list = []
        self.precision_list = []
        self.ndcg_list = []
        self.num_hits_list = []
        self.num_gts_list = []

        self.test_recall_list = []
        self.test_precision_list = []
        self.test_ndcg_list = []
        self.test_num_hits_list = []
        self.test_num_gts_list = []

        self.top_k_list = top_k_list

        list_len = len(self.top_k_list)
        for i in range(list_len):
            self.recall_list.append(0.0)
            self.precision_list.append(0.0)
            self.ndcg_list.append(0.0)
            self.num_hits_list.append(0)
            self.num_gts_list.append(0)
            self.test_recall_list.append(0.0)
            self.test_precision_list.append(0.0)
            self.test_ndcg_list.append(0.0)
            self.test_num_hits_list.append(0)
            self.test_num_gts_list.append(0)

    def update_best_records_based_xth_recall(self, epoch, precision_list, recall_list, ndcg_list,
                                             num_hits_list, num_gts_list, x_th=2, n_step=-1):
        updated = False
        max_recall = 0.0
        cur_recall = 0.0
        for i in range(0, x_th):
            max_recall += self.recall_list[i]
            cur_recall += recall_list[i]

        if max_recall < cur_recall:
            list_len = len(self.top_k_list)
            self.epoch = epoch
            if n_step > -1:
                self.n_step = n_step
            for i in range(list_len):
                self.recall_list[i] = recall_list[i]
                self.ndcg_list[i] = ndcg_list[i]
                self.precision_list[i] = precision_list[i]
                self.num_hits_list[i] = num_hits_list[i]
                self.num_gts_list[i] = num_gts_list[i]
            updated = True
        return updated

    def update_best_records_based_recall_sum(self, epoch, precision_list, recall_list, ndcg_list,
                                             num_hits_list, num_gts_list, n_step=-1):
        updated = False
        if np.sum(self.recall_list) < np.sum(recall_list):
            list_len = len(self.top_k_list)
            self.epoch = epoch
            if n_step > -1:
                self.n_step = n_step
            for i in range(list_len):
                self.recall_list[i] = recall_list[i]
                self.precision_list[i] = precision_list[i]
                self.ndcg_list[i] = ndcg_list[i]
                self.num_hits_list[i] = num_hits_list[i]
                self.num_gts_list[i] = num_gts_list[i]
            updated = True
        return updated

    def update_test_records_when_best_records_got(self, precision_list, recall_list, ndcg_list,
                                                  num_hits_list, num_gts_list):
        list_len = len(self.top_k_list)
        for i in range(list_len):
            self.test_recall_list[i] = recall_list[i]
            self.test_precision_list[i] = precision_list[i]
            self.test_ndcg_list[i] = ndcg_list[i]
            self.test_num_hits_list[i] = num_hits_list[i]
            self.test_num_gts_list[i] = num_gts_list[i]

    def print_best_record(self):
        if self.n_step > -1:
            print("[BEST RESULT] epoch %d, n_step %d" % (self.epoch, self.n_step))
        else:
            print("[BEST RESULT] epoch %d." % self.epoch)

        print("k\tPrec\tRec\tNdcg\tHit\tGt")
        for i, k in enumerate(self.top_k_list):
            print("%d\t%.3f\t%.3f\t%.3f\t%d\t%d"
                  % (k, self.precision_list[i], self.recall_list[i],
                     self.ndcg_list[i],
                     self.num_hits_list[i], self.num_gts_list[i]))

        if self.n_step > -1:
            print("[Test RESULT WHEN BEST RESULT] epoch %d, n_step %d" % (self.epoch, self.n_step))
        else:
            print("[Test RESULT WHEN BEST RESULT] epoch %d." % self.epoch)
        print("k\tPrec\tRec\tNdcg\tHit\tGt")
        for i, k in enumerate(self.top_k_list):
            print("%d\t%.3f\t%.3f\t%.3f\t%d\t%d"
                  % (k, self.test_precision_list[i], self.test_recall_list[i],
                     self.test_ndcg_list[i],
                     self.test_num_hits_list[i], self.test_num_gts_list[i]))
