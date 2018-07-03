# params config

# model config

# feature config
# 'hour_click_ratio', 'period_click_ratio', 'click_ratio'， 'playing_duration_favor'
user_action_features = ['click_ratio', 'browse_num', 'click_num', 'like_num', 'follow_num', 'playing_sum',
                        'duration_sum', 'like_ratio', 'follow_ratio', 'playing_ratio', 'browse_time_diff', 'click_freq',
                        'browse_freq', 'playing_freq', 'duration_favor']

face_features = ['face_num', 'man_num', 'woman_num', 'man_scale', 'woman_scale', 'human_scale', 'man_avg_age',
                 'woman_avg_age', 'human_avg_age', 'man_avg_attr', 'woman_avg_attr', 'human_avg_attr',
                 'woman_num_ratio', 'man_num_ratio']

user_face_favor_features = ['man_favor', 'woman_favor', 'man_cv_favor', 'woman_cv_favor', 'man_age_favor',
                            'woman_age_favor', 'man_yen_value_favor', 'woman_yen_value_favor', 'face_click_favor',
                            'non_face_click_favor']

user_text_favor_features = ['cover_length_favor']

id_features = ['user_id', 'photo_id']
time_features = ['time', 'duration_time']
# time_features = ['time_cate', 'duration_time_cate']


# text_features = ['cover_length', 'avg_tfidf', 'key_words_num','text_class_label','text_cluster_label']
text_features = ['cover_length', 'avg_tfidf', 'key_words_num']

# 'clicked_ratio', 'have_face_cate'
photo_features = ['exposure_num', 'have_face_cate', 'have_text_cate'] + face_features + text_features
user_features = user_action_features + user_face_favor_features + user_text_favor_features

y_label = ['click']

features_to_train = user_features + photo_features + time_features
# features_to_train = list(set(features_to_train) - set(['clicked_ratio']))

cate_features_to_train = []

norm_features = ['browse_num', 'click_num', 'like_num', 'follow_num', 'playing_sum', 'duration_sum', 'click_ratio',
                 'like_ratio', 'follow_ratio', 'playing_ratio', 'browse_time_diff', 'click_freq', 'browse_freq',
                 'playing_freq', 'man_favor', 'woman_favor', 'man_cv_favor', 'woman_cv_favor', 'man_age_favor',
                 'woman_age_favor', 'man_yen_value_favor', 'woman_yen_value_favor', 'face_click_favor',
                 'non_face_click_favor', 'cover_length_favor', 'exposure_num', 'face_num', 'man_num', 'woman_num',
                 'man_scale', 'woman_scale', 'human_scale', 'man_avg_age', 'woman_avg_age', 'human_avg_age',
                 'man_avg_attr', 'woman_avg_attr', 'human_avg_attr', 'cover_length', 'avg_tfidf', 'key_words_num',
                 'time', 'duration_time', 'period_click_ratio', 'playing_favor', 'duration_favor',
                 'playing_duration_favor']

uint64_cols = ['user_id', 'photo_id', 'time']
uint32_cols = ['playing_sum', 'browse_time_diff', 'duration_sum', 'key_words_num']
uint16_cols = ['browse_num', 'exposure_num', 'click_num', 'duration_time', 'like_num', 'follow_num', 'clicked_num']
uint8_cols = ['cover_length', 'man_num', 'woman_num', 'face_num', 'time_cate', 'duration_time_cate']
bool_cols = ['have_face_cate', 'have_text_cate', 'click']
float32_cols = ['period_click_ratio', 'clicked_ratio', 'non_face_click_favor', 'face_click_favor', 'man_favor',
                'woman_avg_age', 'playing_freq', 'woman_age_favor', 'woman_yen_value_favor', 'human_scale',
                'woman_favor', 'click_freq', 'woman_cv_favor', 'man_age_favor', 'man_yen_value_favor', 'follow_ratio',
                'man_scale', 'browse_freq', 'man_avg_age', 'man_cv_favor', 'man_avg_attr', 'playing_ratio',
                'woman_scale', 'click_ratio', 'human_avg_age', 'woman_avg_attr', 'like_ratio', 'cover_length_favor',
                'human_avg_attr', 'avg_tfidf', 'hour_click_ratio', 'woman_num_ratio', 'man_num_ratio', 'playing_favor',
                'duration_favor', 'playing_duration_favor']
float64_cols = []

uint32_cate_cols = []
uint16_cate_cols = []
uint8_cate_cols = ['browse_num_cate', 'click_num_cate', 'like_num_cate', 'follow_num_cate', 'playing_sum_cate',
                   'duration_sum_cate',
                   'click_ratio_cate', 'like_ratio_cate', 'follow_ratio_cate', 'playing_ratio_cate',
                   'browse_time_diff_cate',
                   'click_freq_cate', 'browse_freq_cate', 'playing_freq_cate', 'man_favor_cate', 'woman_favor_cate',
                   'man_cv_favor_cate',
                   'woman_cv_favor_cate', 'man_age_favor_cate', 'woman_age_favor_cate', 'man_yen_value_favor_cate',
                   'woman_yen_value_favor_cate', 'face_click_favor_cate', 'non_face_click_favor_cate',
                   'cover_length_favor_cate',
                   'exposure_num_cate', 'face_num_cate', 'man_num_cate', 'woman_num_cate', 'man_scale_cate',
                   'woman_scale_cate',
                   'human_scale_cate', 'man_avg_age_cate', 'woman_avg_age_cate', 'human_avg_age_cate',
                   'man_avg_attr_cate',
                   'woman_avg_attr_cate', 'human_avg_attr_cate', 'cover_length_cate', 'avg_tfidf_cate',
                   'key_words_num_cate',
                   'time_cate', 'duration_time_cate']
# uint8_cate_cols = ['time_cate', 'duration_time_cate', 'browse_num_cate', 'click_num_cate', 'like_num_cate',
#                     'follow_num_cate', 'playing_sum_cate', 'duration_sum_cate','exposure_num_cate', 'face_num_cate',
#                     'man_num_cate', 'woman_num_cate', 'cover_length_cate', 'key_words_num_cate']

feature_dtype_map = {}
for name in uint64_cols:
    feature_dtype_map.update({name: 'uint64'})
for name in uint32_cols:
    feature_dtype_map.update({name: 'uint32'})
for name in uint16_cols:
    feature_dtype_map.update({name: 'uint16'})
for name in (uint8_cols + uint8_cate_cols):
    feature_dtype_map.update({name: 'uint8'})
for name in bool_cols:
    feature_dtype_map.update({name: 'bool'})
for name in float32_cols:
    feature_dtype_map.update({name: 'float32'})
for name in float64_cols:
    feature_dtype_map.update({name: 'float64'})

# BayesianSmoothing parameters, this already trained
alpha = 2.5171267342473382
beta = 7.087836849232511