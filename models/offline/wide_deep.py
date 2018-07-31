

import os
import sys
import logging
sys.path.append('../../')
from multiprocessing import cpu_count
import argparse
import time


import tensorflow as tf
from common.base import *
from common.utils import load_config_from_pyfile, read_data, normalize_min_max
from conf.modelconf import *

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-v', '--version',
                    help='model version, there will be a version control and a json description file for this model',
                    required=True)
parser.add_argument('-d', '--description', help='description for a model, a json description file attached to a model',
                    required=True)
parser.add_argument('-a', '--all', help='use one ensemble table all, or merge by columns', action='store_true')
parser.add_argument('-n', '--num-workers', help='num used to merge columns', default=cpu_count())
parser.add_argument('-c', '--config-file', help='model config file', default='')
parser.add_argument('-g', '--gpu-mode', help='use gpu mode or not', action="store_true")

args = parser.parse_args()

if __name__ == '__main__':

    fmt = args.format if args.format else 'csv'
    version = args.version
    desc = args.description
    all_one = args.all
    kfold = 0
    num_workers = args.num_workers
    config = load_config_from_pyfile(args.config_file)
    features_to_train = config.features_to_train
    id_features = config.id_features
    user_features = config.user_features
    photo_features = config.photo_features
    time_features = config.time_features
    one_ctr_features = config.one_ctr_features
    combine_ctr_features = config.combine_ctr_features
    y_label = config.y_label

    logging.info('--------------------version: %s ----------------------' % version)
    logging.info('desc: %s' % desc)
    model_name = 'widedeep'

    model_store_path = './sample/' if USE_SAMPLE else './data'

    feature_store_dir = os.path.join(offline_data_dir, 'features')
    col_feature_store_dir = os.path.join(feature_store_dir, 'columns')

    model = Classifier(None, dir=model_store_path, name=model_name, version=version, description=desc,
                       features_to_train=features_to_train)

    start = time.time()
    if all_one:
        ALL_FEATURE_TRAIN_FILE = 'ensemble_feature_train' + str(kfold) + '.' + fmt
        ensemble_train = read_data(os.path.join(feature_store_dir, ALL_FEATURE_TRAIN_FILE), fmt)

        ALL_FEATURE_TEST_FILE = 'ensemble_feature_test' + str(kfold) + '.' + fmt
        ensemble_test = read_data(os.path.join(feature_store_dir, ALL_FEATURE_TEST_FILE), fmt)
    else:
        feature_to_use = id_features + user_features + photo_features + time_features + one_ctr_features + combine_ctr_features
        fm_trainer = FeatureMerger(col_feature_store_dir, feature_to_use + y_label, fmt=fmt, data_type='train',
                                   pool_type='process', num_workers=num_workers)
        fm_tester = FeatureMerger(col_feature_store_dir, feature_to_use + y_label, fmt=fmt, data_type='test',
                                  pool_type='process', num_workers=num_workers)
        ensemble_train = fm_trainer.concat()
        ensemble_test = fm_tester.concat()

    end = time.time()
    print('data read in %s seconds' % str(end - start))

    print(ensemble_train.info())
    print(ensemble_test.info())

    # Categorical base columns.
    categorical_columns = ['gender', 'photo_cluster_label', 'text_cluster_label']
    gender = tf.feature_column.categorical_column_with_vocabulary_list('gender', [0, 1, -1], dtype=tf.int8, default_value=-1)
    photo_cluster_label = tf.feature_column.categorical_column_with_vocabulary_list('photo_cluster_label', [i for i in range(50)] + [-1], dtype=tf.int8, default_value=-1)
    text_cluster_label = tf.feature_column.categorical_column_with_vocabulary_list('text_cluster_label', [i for i in range(20)] + [-1], dtype=tf.int8, default_value=-1)
    # gender = tf.feature_column.indicator_column(gender)
    # photo_cluster_label = tf.feature_column.indicator_column(photo_cluster_label)
    # text_cluster_label = tf.feature_column.indicator_column(text_cluster_label)
    # time = tf.feature_column.numeric_column("time")


    numeric_columns = ['key_words_num', 'exposure_num', 'browse_num', 'face_num', 'woman_num', 'man_num', 'gender', 'age', 'appearance', 'cover_length', 'duration_time']

    # Continuous base columns.
    real_valued_columns = []
    reals = []
    # playing_freq click_freq browse_time_diff browse_freq， not valid, remove them
    float32_cols = one_ctr_features + combine_ctr_features + ['non_face_click_favor', 'face_click_favor',
                'man_favor', 'woman_avg_age', 'woman_age_favor', 'woman_yen_value_favor',
                'human_scale', 'woman_favor', 'woman_cv_favor', 'man_age_favor', 'man_yen_value_favor',
                'follow_ratio', 'man_scale', 'man_avg_age', 'man_cv_favor', 'man_avg_attr',
                'playing_ratio', 'woman_scale', 'click_ratio', 'human_avg_age', 'woman_avg_attr', 'like_ratio',
                'cover_length_favor', 'human_avg_attr', 'avg_tfidf', 'woman_num_ratio',
                'man_num_ratio', 'playing_favor', 'duration_favor', 'playing_duration_favor',
                'face_favor', 'text_clicked_ratio', 'scale']

    features_to_train_sets = set(features_to_train)
    for feat in float32_cols:
        if feat in features_to_train_sets:
            col = tf.feature_column.numeric_column(feat)
            real_valued_columns.append(col)
            numeric_columns.append(feat)
            reals.append(feat)

    print(real_valued_columns)
    num_train = ensemble_train.shape[0]
    data = pd.concat([ensemble_train, ensemble_test])
    normalize_min_max(data, reals)
    ensemble_train = data.iloc[:num_train]
    ensemble_test = data.iloc[num_train:]

    # Continuous base columns.
    key_words_num = tf.feature_column.numeric_column("key_words_num")
    exposure_num = tf.feature_column.numeric_column("exposure_num")
    browse_num = tf.feature_column.numeric_column("browse_num")
    face_num = tf.feature_column.numeric_column("face_num")
    man_num = tf.feature_column.numeric_column("woman_num")
    woman_num = tf.feature_column.numeric_column("man_num")
    age = tf.feature_column.numeric_column("age")
    appearance = tf.feature_column.numeric_column("appearance")
    cover_length = tf.feature_column.numeric_column("cover_length")
    duration_time = tf.feature_column.numeric_column("duration_time")

    key_words_num_buckets = tf.feature_column.bucketized_column(key_words_num, boundaries=[0,1,5])
    exposure_num_buckets = tf.feature_column.bucketized_column(exposure_num, boundaries=[10,100,1000])
    browse_num_buckets = tf.feature_column.bucketized_column(browse_num, boundaries=[100,200,500,1000])
    face_num_buckets = tf.feature_column.bucketized_column(face_num, boundaries=[1,2,3,4])
    woman_num_buckets = tf.feature_column.bucketized_column(woman_num, boundaries=[1,2,3])
    man_num_buckets = tf.feature_column.bucketized_column(man_num, boundaries=[1,2,3])
    age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    appearance_buckets = tf.feature_column.bucketized_column(appearance, boundaries=[60, 70, 80, 90])
    cover_length_buckets = tf.feature_column.bucketized_column(cover_length, boundaries=[10,20])
    duration_time_buckets = tf.feature_column.bucketized_column(duration_time, boundaries=[30, 60, 300])

    wide_buckets_columns = [key_words_num_buckets, exposure_num_buckets, browse_num_buckets, face_num_buckets,
                            woman_num_buckets, man_num_buckets, age_buckets, appearance_buckets, cover_length_buckets, duration_time_buckets]


    vector_feature_columns = ['cover_words', 'pos_photo_id', 'neg_photo_id', 'pos_photo_cluster_label', 'neg_photo_cluster_label', 'pos_user_id', 'neg_user_id']


    wide_columns = [
      gender, photo_cluster_label, text_cluster_label,
      tf.feature_column.crossed_column([gender, photo_cluster_label], hash_bucket_size=int(1e3)),
      tf.feature_column.crossed_column([gender, text_cluster_label], hash_bucket_size=int(1e3)),
      tf.feature_column.crossed_column([photo_cluster_label, text_cluster_label], hash_bucket_size=int(1e4)),
      tf.feature_column.crossed_column([age_buckets, appearance_buckets], hash_bucket_size=int(1e3))] + wide_buckets_columns


    deep_columns = [
      tf.feature_column.embedding_column(gender, dimension=8),
      tf.feature_column.embedding_column(age_buckets, dimension=8),
      tf.feature_column.embedding_column(appearance_buckets, dimension=8),
      tf.feature_column.embedding_column(photo_cluster_label, dimension=8),
      tf.feature_column.embedding_column(text_cluster_label, dimension=4),
    ] + real_valued_columns



    import tempfile
    model_dir = tempfile.mkdtemp()
    model = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50])

    # Read the training and test data sets into Pandas dataframe.

    LABEL_COLUMN = y_label

    def input_fn(df):
      # Creates a dictionary mapping from each continuous feature column name (k) to
      # the values of that column stored in a constant Tensor.
      continuous_features = {k: tf.constant(df[k].values)
                         for k in numeric_columns}
      # Creates a dictionary mapping from each categorical feature column name (k)
      # to the values of that column stored in a tf.SparseTensor.

      categorical_features = {k: tf.SparseTensor(
          indices=[[i, 0] for i in range(df[k].size)],
          values=df[k].values,
          dense_shape=[df[k].size, 1])
                          for k in categorical_columns}
      # categorical_features = {}
      # Merges the two dictionaries into one.
      feature_cols = dict(continuous_features.items() + categorical_features.items())
      # Converts the label column into a constant Tensor.
      label = tf.constant(df[LABEL_COLUMN].values)
      # Returns the feature columns and the label.
      return feature_cols, label

    def train_input_fn():
      return input_fn(ensemble_train)

    def eval_input_fn():
      return input_fn(ensemble_test)

    max_steps = 200
    for i in range(max_steps):
        model.train(input_fn=train_input_fn, steps=1)
        # 评估auc，准确率
        results = model.evaluate(input_fn=eval_input_fn, steps=1)
        for key in sorted(results):
            print("%s: %s" % (key, results[key]))
        # 评估打分
        results = model.predict(input_fn=input_fn('evaluate.csv'))
        for y, r in zip(ensemble_test[y_label].values, results):
            print ('label = %d, positive_ratio = %f'
                   % (y, r['probabilities'][1]))
        print("auc", roc_auc_score(ensemble_test[y_label].values, results['probabilities'][1]))



    # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
    # for n in range(train_epochs // epochs_per_eval):
    #   model.train(input_fn=lambda: input_fn(
    #       train_data, epochs_per_eval, True, batch_size))
    #   results = model.evaluate(input_fn=lambda: input_fn(
    #       FLAGS.test_data, 1, False, FLAGS.batch_size))
    #   # Display evaluation metrics
    #   print('Results at epoch', (n + 1) * epochs_per_eval)
    #   print('-' * 30)
    #   for key in sorted(results):
#     print('%s: %s' % (key, results[key]))