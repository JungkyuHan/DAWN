import tensorflow as tf
import numpy as np
from U_EvalMetricCalculator import EvalMetricCalculator
from U_RecordHolder import RecordHolder
from tqdm import tqdm
import os.path
from U_CustomLayers import CustomSchedule
from U_CustomLayers import DenseBatchLayer
from U_CustomLayers import FCLayer


class NCIR:
    def __init__(self, cf_vec_rank, item_content_vec_rank,
                 attention_vec_dim,
                 model_select, num_neighbors, num_experts,
                 learning_rate, lr_decay, lr_decay_step, cf_vec_dropout,
                 cf_vec_pred_loss_w, layer_weight_reg,
                 data_set_name, model_dir, trials, seed=0):
        self.cf_vec_rank = cf_vec_rank
        self.item_content_vec_rank = item_content_vec_rank
        self.model_select = model_select
        self.num_neighbors = num_neighbors
        self.num_experts = num_experts
        self.trials = trials
        self.seed = seed
        self.pred_loss = tf.keras.losses.MSE
        # optimizer
        self.lr = learning_rate
        self.lr_decay = lr_decay
        self.lr_decay_step = lr_decay_step
        self.lr_schedule = CustomSchedule(initial_lr=learning_rate, decay=lr_decay, decay_step=lr_decay_step)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule, momentum=0.9)
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.is_training = True
        self.csr_dawn = None
        self.user_predictor = None
        self.item_predictor = None
        self.warm_item_transformer = None
        self.diff_based_item_predictor = None
        self.diff_item_cf_predictor = None
        self.content_item_cf_predictor = None

        self.cf_vec_dropout = cf_vec_dropout

        self.attention_vec_dim = attention_vec_dim
        self.cf_vec_pred_loss = tf.keras.losses.MSE
        self.cf_vec_pred_loss_w = cf_vec_pred_loss_w
        self.rating_pred_loss = tf.keras.losses.MSE
        self.layer_weight_reg = layer_weight_reg
        self.model_name = None
        self.data_set_name = data_set_name
        self.model_dir = model_dir

    def print_configurations(self):
        print("/**********************[ALL Runtime Configurations]********************/")
        print("trial: %d" % self.trials)
        print("seed: %d" % self.seed)
        print("cf_vec_rank: %d" % self.cf_vec_rank)
        print("item_content_vec_rank: %d" % self.item_content_vec_rank)
        print("model_select: ")
        print(self.model_select)
        print("num_neighbors: %d" % self.num_neighbors)
        print("num_experts: %d" % self.num_experts)
        print("attention_vec_dim: %d" % self.attention_vec_dim)
        print("cf_vec_dropout: %.2f" % self.cf_vec_dropout)
        print("cf_vec_pred_loss_w: %f" % self.cf_vec_pred_loss_w)
        print("layer_weight_reg: %f" % self.layer_weight_reg)
        print("learn rate: %f" % self.lr)
        print("learn rate decay: %f" % self.lr_decay)
        print("learn rate decay step: %d" % self.lr_decay_step)
        print("/**********************************************************************/")

    def get_model_name(self):
        model_name = "{}_{}_{}_{}_{}_{}_{}".format(self.model_name, self.data_set_name,
                                                self.cf_vec_dropout, self.attention_vec_dim,
                                                self.num_experts, self.num_neighbors,
                                                self.trials)
        return model_name

    def get_model_full_path(self):
        model_file_name = self.get_model_name()
        model_dir = model_file_name + "/"
        complete_file_path = os.path.join(self.model_dir, model_dir)
        complete_file_path = os.path.join(complete_file_path, model_file_name)
        return complete_file_path

    def build_model(self, num_heads=1):
        self.model_name = "CSR_Dawn"
        num_neighbors = self.num_neighbors

        u_in = tf.keras.Input(shape=[self.cf_vec_rank, ], name="in_u_cf", dtype=tf.float32)
        v_in = tf.keras.Input(shape=[self.cf_vec_rank, ], name="in_v_cf", dtype=tf.float32)
        dropout_indicator = tf.keras.Input(shape=(1,), name='dropout_indicator', dtype=tf.dtypes.float32)

        ############## Item Cold Start

        v_content_in = tf.keras.Input(shape=[self.item_content_vec_rank, ], name="in_v_content", dtype=tf.float32)

        v_in_neighbors = []
        v_content_in_neighbors = []
        v_content_in_diff_neighbors = []
        for i in range(num_neighbors):
            v_in_n = tf.keras.Input(shape=[self.cf_vec_rank, ], dtype=tf.float32)
            v_in_neighbors.append(v_in_n)
            v_content_in_n = tf.keras.Input(shape=[self.item_content_vec_rank, ], dtype=tf.float32)
            v_content_in_neighbors.append(v_content_in_n)
            contents_diff = v_content_in - v_content_in_n
            v_content_in_diff_neighbors.append(contents_diff)

        ##### Heater : CF Vector prediction from Content Vector
        num_experts = self.num_experts
        item_content_gate = FCLayer(units=num_experts, use_activation=True,
                                    regularizer_weight=self.layer_weight_reg)(v_content_in)
        item_content_expert_list = []
        for i in range(num_experts):
            tmp_expert = v_content_in
            for h_layer_idx, num_hidden_units in enumerate(self.model_select):
                tmp_expert = FCLayer(units=num_hidden_units, use_activation=True,
                                     regularizer_weight=self.layer_weight_reg)(tmp_expert)
            item_content_expert_list.append(tf.reshape(tmp_expert, [-1, 1, self.cf_vec_rank]))
        # size: num_experts X self.output_rank
        item_content_expert_concat = tf.concat(item_content_expert_list, 1)
        item_content_expert_concat = tf.linalg.matmul(
            tf.reshape(item_content_gate, [-1, 1, num_experts]), item_content_expert_concat)
        # size: batch_size X self.output_rank
        v_cf_vec_predicted = tf.reshape(tf.nn.tanh(item_content_expert_concat), [-1, self.cf_vec_rank])

        # item_cf_predict_list = [v_cf_vec_predicted]
        ######### SimCSRDiff : CF Vector diff prediction from Content Vector
        cf_diff_prediction = FCLayer(units=self.cf_vec_rank, use_activation=True,
                                     regularizer_weight=self.layer_weight_reg)
        item_cf_pred_from_diff_list = []
        for i_n, diff_neighbors in enumerate(v_content_in_diff_neighbors):
            diff_input = diff_neighbors
            cf_diff_pred = cf_diff_prediction(diff_input)
            cf_vec_pred_from_diff = v_in_neighbors[i_n] + cf_diff_pred
            item_cf_pred_from_diff_list.append(cf_vec_pred_from_diff)

        v_cf_vec_predicted_from_diff_avg = tf.reduce_mean(item_cf_pred_from_diff_list, 0)
        v_cf_vec_predicted_from_diff = tf.nn.tanh(v_cf_vec_predicted_from_diff_avg)

        v_cf_predicted = (v_cf_vec_predicted + v_cf_vec_predicted_from_diff) / 2.0

        cf_vec_filter = 1.0 - dropout_indicator
        item_selected_cf = (v_in * cf_vec_filter + v_cf_predicted * dropout_indicator)

        ######### item
        v_mid_layer = item_selected_cf
        v_mid_layer = DenseBatchLayer(units=self.cf_vec_rank, is_training=True,
                                          do_norm=True, regularizer_weight=self.layer_weight_reg)(v_mid_layer)
        v_embedding1 = FCLayer(units=self.cf_vec_rank, use_activation=False,
                              regularizer_weight=self.layer_weight_reg)(v_mid_layer)

        ######### User
        u_mid_layer = u_in
        u_mid_layer = DenseBatchLayer(units=self.cf_vec_rank, is_training=True,
                                          do_norm=True, regularizer_weight=self.layer_weight_reg)(u_mid_layer)
        u_embedding = FCLayer(units=self.cf_vec_rank, use_activation=False,
                              regularizer_weight=self.layer_weight_reg)(u_mid_layer)

        attention_dim = self.attention_vec_dim
        l2_regularizer = tf.keras.regularizers.l2(self.layer_weight_reg)
        truncated_normal_init = tf.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None)
        v_n_embedding_list = []

        for i in range(num_heads):
            ga_Q = tf.keras.layers.Dense(units=attention_dim, activation=None, use_bias=False,
                                         kernel_initializer=truncated_normal_init,
                                         kernel_regularizer=l2_regularizer,
                                         activity_regularizer=None, kernel_constraint=None,
                                         bias_constraint=None)
            ga_W = tf.keras.layers.Dense(units=attention_dim, activation=None, use_bias=False,
                                         kernel_initializer=truncated_normal_init,
                                         kernel_regularizer=l2_regularizer,
                                         activity_regularizer=None, kernel_constraint=None,
                                         bias_constraint=None)
            ga_Val = tf.keras.layers.Dense(units=attention_dim, activation=None, use_bias=False,
                                           kernel_initializer=truncated_normal_init,
                                           kernel_regularizer=l2_regularizer,
                                           activity_regularizer=None, kernel_constraint=None,
                                           bias_constraint=None)
            ga_a = tf.keras.layers.Dense(units=1, activation=tf.nn.leaky_relu, use_bias=False,
                                         kernel_initializer=truncated_normal_init,
                                         kernel_regularizer=l2_regularizer,
                                         activity_regularizer=None, kernel_constraint=None,
                                         bias_constraint=None)
            item_value_list = []
            sim_input_list = []

            item_query_tgt_t = ga_Q(v_content_in)

            for j in range(num_neighbors):
                v_neighbor = v_in_neighbors[j]
                item_key_neighbor_t = ga_W(v_neighbor)
                sim_input = tf.concat([item_query_tgt_t, item_key_neighbor_t], 1)
                sim_input = ga_a(sim_input)
                sim_input_list.append(sim_input)
                v_neighbor_input = tf.concat([item_query_tgt_t, item_key_neighbor_t, v_neighbor], 1)
                item_neighbor_t = ga_Val(v_neighbor_input)
                item_neighbor_re = tf.reshape(item_neighbor_t, [-1, 1, attention_dim])
                item_value_list.append(item_neighbor_re)

            sim_input_list = tf.concat(sim_input_list, 1)
            ga_alpha = tf.keras.layers.Softmax()(sim_input_list)

            item_value_concat = tf.concat(item_value_list, 1)
            ga_alpha_mat = tf.reshape(ga_alpha, [-1, 1, num_neighbors])

            v_n_embedding = tf.linalg.matmul(ga_alpha_mat, item_value_concat)
            v_n_embedding = tf.reshape(v_n_embedding, [-1, attention_dim])
            v_n_embedding_list.append(v_n_embedding)
        v_n_final_embedding = tf.math.reduce_mean(v_n_embedding_list, 0)

        v_embedding = v_embedding1 + v_n_final_embedding
        ############# SimCSR : end
        prediction = tf.math.multiply(u_embedding, v_embedding)
        prediction = tf.math.reduce_sum(prediction, axis=1, keepdims=True)  # output of the model, the predicted scores

        model_inputs = [u_in, v_in, dropout_indicator]
        model_outputs = [prediction]

        if self.item_content_vec_rank > 0:
            model_inputs.append(v_content_in)
            for in_v in v_in_neighbors:
                model_inputs.append(in_v)
            for in_content_v in v_content_in_neighbors:
                model_inputs.append(in_content_v)
            model_outputs.append(v_cf_vec_predicted)
            model_outputs.append(v_cf_vec_predicted_from_diff)

        self.csr_dawn = tf.keras.Model(inputs=model_inputs, outputs=model_outputs)

        user_predictor_inputs = [u_in]

        item_predictor_inputs = [v_in]
        if self.item_content_vec_rank > 0:
            item_predictor_inputs.append(dropout_indicator)
            item_predictor_inputs.append(v_content_in)
            for in_v in v_in_neighbors:
                item_predictor_inputs.append(in_v)
            for in_content_v in v_content_in_neighbors:
                item_predictor_inputs.append(in_content_v)

        self.user_predictor = tf.keras.Model(inputs=user_predictor_inputs, outputs=u_embedding)
        self.item_predictor = tf.keras.Model(inputs=item_predictor_inputs, outputs=v_embedding)
        self.warm_item_transformer = tf.keras.Model(inputs=item_predictor_inputs, outputs=v_embedding1)
        self.diff_based_item_predictor = tf.keras.Model(inputs=item_predictor_inputs, outputs=v_cf_predicted)
        self.diff_item_cf_predictor = tf.keras.Model(inputs=item_predictor_inputs, outputs=v_cf_vec_predicted_from_diff)
        self.content_item_cf_predictor = tf.keras.Model(inputs=item_predictor_inputs, outputs=v_cf_vec_predicted)

    def save_weights(self, save_path):
        self.csr_dawn.save_weights(save_path)

    def load_weights(self, load_path):
        self.csr_dawn.load_weights(load_path)

    def load_weights_by_config(self):
        self.print_configurations()
        complete_file_path = self.get_model_full_path()
        self.load_weights(complete_file_path)

    @staticmethod
    def negative_sampling(pos_user_array, pos_item_array, neg, item_warm):
        neg = int(neg)
        user_pos = pos_user_array.reshape((-1))
        user_neg = np.tile(pos_user_array, neg).reshape((-1))
        item_pos = pos_item_array.reshape((-1))
        item_neg = np.random.choice(item_warm, size=(neg * pos_user_array.shape[0]), replace=True).reshape((-1))
        target_pos = np.ones_like(item_pos)
        target_neg = np.zeros_like(item_neg)
        return np.concatenate((user_pos, user_neg)), np.concatenate((item_pos, item_neg)), \
               np.concatenate((target_pos, target_neg))

    def fit(self, user_tr_list, item_tr_list,
            u_pref, v_pref, v_content, item_warm, v_neighbor_mat,
            num_negative_samples=5, data_batch_size=1024, dropout=0.0, epochs=1,
            tuning_data=None, test_data=None, eval_top_k_list=[20]):

        self.print_configurations()
        num_neighbors = self.num_neighbors
        print("[Evaluation before train]")
        best_record = RecordHolder(eval_top_k_list)

        if tuning_data is not None:
            precision_list, recall_list, ndcg_list, num_hits_list, num_gts_list \
                = self.test(tuning_data, v_pref, v_content, eval_top_k_list, "Eval")

        for epoch in range(epochs):
            self.lr_schedule.set_epoch(epoch + 1)

            user_array, item_array, target_array = NCIR.negative_sampling(user_tr_list, item_tr_list,
                                                                              num_negative_samples, item_warm)

            random_idx = np.random.permutation(user_array.shape[0])
            n_targets = len(random_idx)
            data_batch = [(n, min(n + data_batch_size, n_targets)) for n in range(0, n_targets, data_batch_size)]

            # variables for loss monitoring -- start
            loss_epoch = 0.
            item_cf_diff_loss_epoch = 0.
            item_cf_diff_diff_loss_epoch = 0.
            user_cf_diff_loss_epoch = 0.
            rating_loss_epoch = 0.
            regularization_loss_epoch = 0.
            regularization_loss_sum = 0.
            # variables for loss monitoring -- end

            gen = data_batch
            gen = tqdm(gen)

            for itr_cnter, (start, stop) in enumerate(gen):
                batch_idx = random_idx[start:stop]
                batch_users = user_array[batch_idx]
                batch_items = item_array[batch_idx]
                batch_targets = target_array[batch_idx]

                # dropout
                if dropout != 0.0:
                    n_to_drop = int(np.floor(dropout * len(batch_idx)))  # number of u-i pairs to be dropped
                    zero_index = np.random.choice(np.arange(len(batch_idx)), n_to_drop, replace=False)
                else:
                    zero_index = np.array([])

                user_cf_batch = u_pref[batch_users, :]
                user_cf_batch_tf = tf.convert_to_tensor(user_cf_batch)

                item_cf_batch = v_pref[batch_items, :]
                item_cf_batch_tf = tf.convert_to_tensor(item_cf_batch)

                item_content_batch = v_content[batch_items, :]
                item_content_batch_tf = tf.convert_to_tensor(item_content_batch)

                target_batch_tf = tf.convert_to_tensor(batch_targets)
                num_targets = tf.shape(target_batch_tf)[0]
                target_batch_tf = tf.reshape(target_batch_tf, shape=[num_targets, 1])
                target_batch_tf = tf.cast(x=target_batch_tf, dtype=tf.float32)

                dropout_indicator = np.zeros_like(batch_targets).reshape((-1, 1))
                if len(zero_index) > 0:
                    dropout_indicator[zero_index] = 1.0
                dropout_indicator_tf = tf.convert_to_tensor(dropout_indicator, dtype=tf.float32)


                # neighbor item cf vectors'\
                num_max_neighbors = v_neighbor_mat.shape[1]
                num_neighbors = min(num_max_neighbors, num_neighbors)
                item_neighbors = v_neighbor_mat[batch_items, 0:num_neighbors]  # batch_size * num_neighbors

                v_target_neighbors_cf_tf = []
                for n_idx in range(num_neighbors):
                    neighbors = item_neighbors[:, n_idx]
                    v_neighbor_cf = v_pref[neighbors, :]
                    v_neighbor_cf_tf = tf.convert_to_tensor(v_neighbor_cf)
                    v_target_neighbors_cf_tf.append(v_neighbor_cf_tf)

                # neighbor item content vectors'\
                v_target_neighbors_content_tf = []
                for n_idx in range(num_neighbors):
                    neighbors = item_neighbors[:, n_idx]
                    v_neighbor_content = v_content[neighbors, :]
                    v_neighbor_content = tf.convert_to_tensor(v_neighbor_content)
                    v_target_neighbors_content_tf.append(v_neighbor_content)


                train_inputs = [user_cf_batch_tf, item_cf_batch_tf, dropout_indicator_tf]

                train_inputs.append(item_content_batch_tf)
                train_inputs.append(v_target_neighbors_cf_tf)
                train_inputs.append(v_target_neighbors_content_tf)

                with tf.GradientTape() as tape:
                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    rating_predictions, item_cf_predictions, item_cf_predictions_diff \
                        = self.csr_dawn(train_inputs, training=True)

                    regularization_loss = tf.math.add_n(self.csr_dawn.losses)
                    user_cf_prediction_loss_o = 0.0

                    item_cf_prediction_loss_o = tf.math.squared_difference(item_cf_batch_tf, item_cf_predictions)
                    item_cf_prediction_loss_o = tf.math.reduce_sum(item_cf_prediction_loss_o, axis=1, keepdims=True)
                    item_cf_prediction_loss_o = tf.math.reduce_sum(item_cf_prediction_loss_o)
                    item_cf_prediction_loss_o = self.cf_vec_pred_loss_w * item_cf_prediction_loss_o

                    item_cf_prediction_diff_loss_o \
                        = tf.math.squared_difference(item_cf_batch_tf, item_cf_predictions_diff)
                    item_cf_prediction_diff_loss_o = tf.math.reduce_sum(item_cf_prediction_diff_loss_o, axis=1,
                                                                        keepdims=True)
                    item_cf_prediction_diff_loss_o = tf.math.reduce_sum(item_cf_prediction_diff_loss_o)
                    item_cf_prediction_diff_loss_o = self.cf_vec_pred_loss_w * item_cf_prediction_diff_loss_o

                    rating_prediction_loss_o = tf.math.reduce_mean(
                        tf.math.squared_difference(target_batch_tf, rating_predictions))

                    weighted_loss = item_cf_prediction_loss_o + item_cf_prediction_diff_loss_o \
                                    + rating_prediction_loss_o + regularization_loss

                gradients = tape.gradient(weighted_loss,
                                          self.csr_dawn.trainable_weights)

                self.optimizer.apply_gradients(zip(gradients, self.csr_dawn.trainable_weights))

                rating_loss_epoch += rating_prediction_loss_o
                item_cf_diff_loss_epoch += item_cf_prediction_loss_o
                item_cf_diff_diff_loss_epoch += item_cf_prediction_diff_loss_o
                user_cf_diff_loss_epoch += user_cf_prediction_loss_o
                regularization_loss_sum += regularization_loss  # because this loss is related to layer weights
                regularization_loss_epoch = regularization_loss_sum / itr_cnter
                weighted_loss_epoch = rating_prediction_loss_o + user_cf_prediction_loss_o + item_cf_prediction_loss_o \
                                      + item_cf_prediction_diff_loss_o
                loss_epoch += weighted_loss_epoch

            print("loss sum in epoch %d/%d - all: %.3f, rating: %.3f, ucf: %.3f, icf: %.3f, icf_diff: %.3f, reg: %.3f"
                  % (epoch, epochs, loss_epoch, rating_loss_epoch, user_cf_diff_loss_epoch,
                     item_cf_diff_loss_epoch, item_cf_diff_diff_loss_epoch, regularization_loss_epoch))

            if tuning_data is not None:
                precision_list, recall_list, ndcg_list, num_hits_list, num_gts_list \
                    = self.test(tuning_data, v_pref, v_content, eval_top_k_list, "Eval")
                updated = best_record.update_best_records_based_xth_recall(epoch, precision_list, recall_list,
                                                                           ndcg_list, num_hits_list, num_gts_list)
                if updated:
                    complete_file_path = self.get_model_full_path()
                    self.save_weights(complete_file_path)

                if updated and test_data is not None:
                    test_precision_list, test_recall_list, test_ndcg_list, test_num_hits_list, test_num_gts_list \
                        = self.test(test_data, v_pref, v_content, eval_top_k_list, "Test")
                    best_record.update_test_records_when_best_records_got(test_precision_list, test_recall_list,
                                                                          test_ndcg_list,
                                                                          test_num_hits_list, test_num_gts_list)
                best_record.print_best_record()

        if tuning_data is not None and test_data is not None:
            best_record.print_best_record()

    def load_and_test(self, test_data, v_pref, v_content, eval_top_k_list=[20]):
        self.print_configurations()
        complete_file_path = self.get_model_full_path()
        self.load_weights(complete_file_path)

        test_precision_list, test_recall_list, test_ndcg_list, test_num_hits_list, test_num_gts_list \
            = self.test(test_data, v_pref, v_content, eval_top_k_list, "Test")
        best_record = RecordHolder(eval_top_k_list)
        best_record.update_test_records_when_best_records_got(test_precision_list, test_recall_list,
                                                              test_ndcg_list,
                                                              test_num_hits_list, test_num_gts_list)
        best_record.print_best_record()

    def predict_4_item_cs(self, u_pref_tf, v_pref_tf, v_content_tf, v_neighbor_cf_list_tf,
                          v_neighbor_content_list_tf, top_k):
        num_items = tf.shape(v_pref_tf)[0]
        item_cf_vec_dropout = np.ones((num_items, 1))
        item_cf_vec_dropout_tf = tf.convert_to_tensor(item_cf_vec_dropout, dtype=tf.float32)

        item_vecs = self.item_predictor([v_pref_tf, item_cf_vec_dropout_tf, v_content_tf,
                                         v_neighbor_cf_list_tf, v_neighbor_content_list_tf], training=False)
        user_vecs = self.user_predictor([u_pref_tf], training=False)

        predicted_ratings = tf.linalg.matmul(user_vecs, item_vecs, transpose_b=True)
        top_k_val_mat, top_k_index_mat = tf.math.top_k(input=predicted_ratings, k=top_k, sorted=True)

        top_k_index_tensor = tf.convert_to_tensor(value=top_k_index_mat, dtype=tf.int32)
        top_k_val_tensor = tf.convert_to_tensor(value=top_k_val_mat, dtype=tf.float32)
        return top_k_index_tensor, top_k_val_tensor

    def test(self, test_data, v_pref, v_content, top_k_list, result_label="Test"):
        metric = EvalMetricCalculator(top_k_list=top_k_list)
        max_top_k = 1
        for k in top_k_list:
            if max_top_k < k:
                max_top_k = k

        v_pref_eval = test_data.V_pref_test
        v_pref_eval_tf = tf.convert_to_tensor(v_pref_eval)

        if self.item_content_vec_rank > 0:
            v_content_eval = test_data.V_content_test
            v_content_eval_tf = tf.convert_to_tensor(v_content_eval)

            v_neighbor_mat = test_data.test_item_neighbors
            num_neighbors = self.num_neighbors
            num_max_neighbors = v_neighbor_mat.shape[1]
            num_neighbors = min(num_max_neighbors, num_neighbors)
            item_neighbors = v_neighbor_mat[:, 0:num_neighbors]  # batch_size * num_neighbors

            v_target_neighbors_cf_tf = []
            for n_idx in range(num_neighbors):
                neighbors = item_neighbors[:, n_idx]
                v_neighbor_cf = v_pref[neighbors, :]
                v_neighbor_cf_tf = tf.convert_to_tensor(v_neighbor_cf)
                v_target_neighbors_cf_tf.append(v_neighbor_cf_tf)

            v_target_neighbors_content_tf = []
            for n_idx in range(num_neighbors):
                neighbors = item_neighbors[:, n_idx]
                v_neighbor_content = v_content[neighbors, :]
                v_neighbor_content_tf = tf.convert_to_tensor(v_neighbor_content)
                v_target_neighbors_content_tf.append(v_neighbor_content_tf)

        for i, (eval_start, eval_finish) in enumerate(test_data.eval_batch):
            u_pref_eval = test_data.U_pref_test[eval_start:eval_finish]
            u_pref_eval_tf = tf.convert_to_tensor(u_pref_eval)

            top_k_index_mat, top_k_val_mat \
                = self.predict_4_item_cs(u_pref_tf=u_pref_eval_tf,
                                         v_pref_tf=v_pref_eval_tf,
                                         v_content_tf=v_content_eval_tf,
                                         v_neighbor_cf_list_tf=v_target_neighbors_cf_tf,
                                         v_neighbor_content_list_tf=v_target_neighbors_content_tf,
                                         top_k=max_top_k)
            metric.collect_predictions(top_k_index_mat=top_k_index_mat)

        precision_list, recall_list, ndcg_list, num_hits_list, num_gts_list \
            = metric.get_metric_distributed(test_data=test_data)
        EvalMetricCalculator.print_test_result(top_k_list, precision_list, recall_list, ndcg_list, num_hits_list,
                                               num_gts_list,
                                               result_label)

        return precision_list, recall_list, ndcg_list, num_hits_list, num_gts_list

    def get_cold_item_vec(self, v_pref_tf, v_content_tf, v_neighbor_cf_list_tf, v_neighbor_content_list_tf):
        num_items = tf.shape(v_pref_tf)[0]
        item_cf_vec_dropout = np.ones((num_items, 1))
        item_cf_vec_dropout_tf = tf.convert_to_tensor(item_cf_vec_dropout, dtype=tf.float32)
        cold_item_vecs = self.item_predictor([v_pref_tf, item_cf_vec_dropout_tf, v_content_tf,
                                         v_neighbor_cf_list_tf, v_neighbor_content_list_tf], training=False)
        return cold_item_vecs

    def get_cold_item_vec_prediction(self, test_data, v_pref, v_content):
        v_pref_eval = test_data.V_pref_test
        v_pref_eval_tf = tf.convert_to_tensor(v_pref_eval)

        v_content_eval = test_data.V_content_test
        v_content_eval_tf = tf.convert_to_tensor(v_content_eval)

        v_neighbor_mat = test_data.test_item_neighbors
        num_neighbors = self.num_neighbors
        num_max_neighbors = v_neighbor_mat.shape[1]
        num_neighbors = min(num_max_neighbors, num_neighbors)
        item_neighbors = v_neighbor_mat[:, 0:num_neighbors]  # batch_size * num_neighbors

        v_target_neighbors_cf_tf = []
        for n_idx in range(num_neighbors):
            neighbors = item_neighbors[:, n_idx]
            v_neighbor_cf = v_pref[neighbors, :]
            v_neighbor_cf_tf = tf.convert_to_tensor(v_neighbor_cf)
            v_target_neighbors_cf_tf.append(v_neighbor_cf_tf)

        v_target_neighbors_content_tf = []
        for n_idx in range(num_neighbors):
            neighbors = item_neighbors[:, n_idx]
            v_neighbor_content = v_content[neighbors, :]
            v_neighbor_content_tf = tf.convert_to_tensor(v_neighbor_content)
            v_target_neighbors_content_tf.append(v_neighbor_content_tf)

        cold_item_vecs = self.get_cold_item_vec(v_pref_eval_tf, v_content_eval_tf,
                                                v_target_neighbors_cf_tf, v_target_neighbors_content_tf)
        return cold_item_vecs, item_neighbors

    def get_warm_item_vec(self, v_pref_tf):
        num_items = tf.shape(v_pref_tf)[0]

        item_cf_vec_not_dropout = np.zeros((num_items, 1))
        item_cf_vec_not_dropout_tf = tf.convert_to_tensor(item_cf_vec_not_dropout, dtype=tf.float32)

        v_content_tf = tf.constant(0.0, dtype=tf.float32, shape=(num_items, self.item_content_vec_rank))
        v_neighbor_cf_list_tf = []
        v_neighbor_content_list_tf = []
        for i in range(0, self.num_neighbors):
            v_n_cf_tf = tf.constant(0.0, dtype=tf.float32, shape=(num_items, self.cf_vec_rank))
            v_neighbor_cf_list_tf.append(v_n_cf_tf)
            v_n_content_tf = tf.constant(0.0, dtype=tf.float32, shape=(num_items, self.item_content_vec_rank))
            v_neighbor_content_list_tf.append(v_n_content_tf)

        warm_item_vecs = self.warm_item_transformer([v_pref_tf, item_cf_vec_not_dropout_tf, v_content_tf,
                                                     v_neighbor_cf_list_tf, v_neighbor_content_list_tf], training=False)

        return warm_item_vecs

    def get_diff_based_cold_item_vec(self, v_pref_tf, v_content_tf, v_neighbor_cf_list_tf, v_neighbor_content_list_tf):
        num_items = tf.shape(v_pref_tf)[0]
        item_cf_vec_dropout = np.ones((num_items, 1))
        item_cf_vec_dropout_tf = tf.convert_to_tensor(item_cf_vec_dropout, dtype=tf.float32)
        cold_item_vecs = self.diff_based_item_predictor([v_pref_tf, item_cf_vec_dropout_tf, v_content_tf,
                                         v_neighbor_cf_list_tf, v_neighbor_content_list_tf], training=False)
        return cold_item_vecs

    def get_diff_based_cold_item_vec_prediction(self, test_data, v_pref, v_content):
        v_pref_eval = test_data.V_pref_test
        v_pref_eval_tf = tf.convert_to_tensor(v_pref_eval)

        v_content_eval = test_data.V_content_test
        v_content_eval_tf = tf.convert_to_tensor(v_content_eval)

        v_neighbor_mat = test_data.test_item_neighbors
        num_neighbors = self.num_neighbors
        num_max_neighbors = v_neighbor_mat.shape[1]
        num_neighbors = min(num_max_neighbors, num_neighbors)
        item_neighbors = v_neighbor_mat[:, 0:num_neighbors]  # batch_size * num_neighbors

        v_target_neighbors_cf_tf = []
        for n_idx in range(num_neighbors):
            neighbors = item_neighbors[:, n_idx]
            v_neighbor_cf = v_pref[neighbors, :]
            v_neighbor_cf_tf = tf.convert_to_tensor(v_neighbor_cf)
            v_target_neighbors_cf_tf.append(v_neighbor_cf_tf)

        v_target_neighbors_content_tf = []
        for n_idx in range(num_neighbors):
            neighbors = item_neighbors[:, n_idx]
            v_neighbor_content = v_content[neighbors, :]
            v_neighbor_content_tf = tf.convert_to_tensor(v_neighbor_content)
            v_target_neighbors_content_tf.append(v_neighbor_content_tf)

        cold_item_vecs = self.get_diff_based_cold_item_vec(v_pref_eval_tf, v_content_eval_tf,
                                                           v_target_neighbors_cf_tf, v_target_neighbors_content_tf)
        return cold_item_vecs, item_neighbors

    def get_item_vec_prediction(self, v_pref, v_content, v_neighbor_mat):
        v_pref_eval = v_pref
        v_pref_eval_tf = tf.convert_to_tensor(v_pref_eval)

        v_content_eval = v_content
        v_content_eval_tf = tf.convert_to_tensor(v_content_eval)

        num_neighbors = self.num_neighbors
        num_max_neighbors = v_neighbor_mat.shape[1]
        num_neighbors = min(num_max_neighbors, num_neighbors)
        item_neighbors = v_neighbor_mat[:, 0:num_neighbors]  # batch_size * num_neighbors

        v_target_neighbors_cf_tf = []
        for n_idx in range(num_neighbors):
            neighbors = item_neighbors[:, n_idx]
            v_neighbor_cf = v_pref[neighbors, :]
            v_neighbor_cf_tf = tf.convert_to_tensor(v_neighbor_cf)
            v_target_neighbors_cf_tf.append(v_neighbor_cf_tf)

        v_target_neighbors_content_tf = []
        for n_idx in range(num_neighbors):
            neighbors = item_neighbors[:, n_idx]
            v_neighbor_content = v_content[neighbors, :]
            v_neighbor_content_tf = tf.convert_to_tensor(v_neighbor_content)
            v_target_neighbors_content_tf.append(v_neighbor_content_tf)

        item_vecs_1st = self.get_diff_based_cold_item_vec(v_pref_eval_tf, v_content_eval_tf,
                                                           v_target_neighbors_cf_tf, v_target_neighbors_content_tf)
        item_vecs_2nd = self.get_cold_item_vec(v_pref_eval_tf, v_content_eval_tf,
                                                           v_target_neighbors_cf_tf, v_target_neighbors_content_tf)
        item_vecs_1st_trans = self.get_diff_based_cold_item_trans_vec(v_pref_eval_tf, v_content_eval_tf,
                                                           v_target_neighbors_cf_tf, v_target_neighbors_content_tf)
        item_vec_diff_cf = self.get_diff_item_cf_vec(v_pref_eval_tf, v_content_eval_tf,
                                                           v_target_neighbors_cf_tf, v_target_neighbors_content_tf)
        item_vec_content_cf = self.get_content_item_cf_vec(v_pref_eval_tf, v_content_eval_tf,
                                                           v_target_neighbors_cf_tf, v_target_neighbors_content_tf)

        item_vec_diff_cf = self.get_warm_item_vec(item_vec_diff_cf)
        item_vec_content_cf = self.get_warm_item_vec(item_vec_content_cf)

        return item_vecs_1st, item_vecs_1st_trans, item_vecs_2nd, item_neighbors, item_vec_diff_cf, item_vec_content_cf

    def get_diff_based_cold_item_trans_vec(self, v_pref_tf, v_content_tf, v_neighbor_cf_list_tf,
                                           v_neighbor_content_list_tf):
        num_items = tf.shape(v_pref_tf)[0]
        item_cf_vec_dropout = np.ones((num_items, 1))
        item_cf_vec_dropout_tf = tf.convert_to_tensor(item_cf_vec_dropout, dtype=tf.float32)
        cold_item_vecs = self.warm_item_transformer([v_pref_tf, item_cf_vec_dropout_tf, v_content_tf,
                                                     v_neighbor_cf_list_tf, v_neighbor_content_list_tf], training=False)
        return cold_item_vecs

    def get_user_vec_prediction(self, u_pref):
        u_pref_tf = tf.convert_to_tensor(u_pref)
        user_vecs_trans = self.user_predictor([u_pref_tf], training=False)
        return u_pref, user_vecs_trans

    def get_diff_item_cf_vec(self, v_pref_tf, v_content_tf, v_neighbor_cf_list_tf, v_neighbor_content_list_tf):
        num_items = tf.shape(v_pref_tf)[0]
        item_cf_vec_dropout = np.ones((num_items, 1))
        item_cf_vec_dropout_tf = tf.convert_to_tensor(item_cf_vec_dropout, dtype=tf.float32)
        cold_item_vecs = self.diff_item_cf_predictor([v_pref_tf, item_cf_vec_dropout_tf, v_content_tf,
                                                     v_neighbor_cf_list_tf, v_neighbor_content_list_tf], training=False)
        return cold_item_vecs

    def get_content_item_cf_vec(self, v_pref_tf, v_content_tf, v_neighbor_cf_list_tf, v_neighbor_content_list_tf):
        num_items = tf.shape(v_pref_tf)[0]
        item_cf_vec_dropout = np.ones((num_items, 1))
        item_cf_vec_dropout_tf = tf.convert_to_tensor(item_cf_vec_dropout, dtype=tf.float32)
        cold_item_vecs = self.content_item_cf_predictor([v_pref_tf, item_cf_vec_dropout_tf, v_content_tf,
                                                        v_neighbor_cf_list_tf, v_neighbor_content_list_tf], training=False)
        return cold_item_vecs
