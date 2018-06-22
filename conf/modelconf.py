#coding:utf8    

import logging

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] [%(levelname)s] %(message)s',
                # datefmt='%Y-%M-%D %H:%M:%S',
                filename='kuaishou.log',
                filemode='w')

#################################################################################################
#定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] [%(levelname)s] %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
#################################################################################################


# params config

# model config

# feature config
# 'hour_click_ratio', 'period_click_ratio'
user_action_features = ['browse_num', 'click_num', 'like_num', 'follow_num', 'playing_sum','duration_sum', 'click_ratio', 'like_ratio', 'follow_ratio', 'playing_ratio', 'browse_time_diff', 'click_freq', 'browse_freq', 'playing_freq']
    
face_features = ['face_num', 'man_num', 'woman_num', 'man_scale', 'woman_scale', 'human_scale', 'man_avg_age', 'woman_avg_age', 'human_avg_age',  'man_avg_attr', 'woman_avg_attr', 'human_avg_attr']
    
    
user_face_favor_features = ['man_favor', 'woman_favor', 'man_cv_favor', 'woman_cv_favor', 'man_age_favor', 'woman_age_favor', 'man_yen_value_favor', 'woman_yen_value_favor', 'face_click_favor', 'non_face_click_favor']

user_text_favor_features = ['cover_length_favor']
    
    
id_features = ['user_id', 'photo_id']
time_features = ['time', 'duration_time']
# time_features = ['time_cate', 'duration_time_cate']


text_features = ['cover_length', 'avg_tfidf', 'key_words_num']

#'clicked_ratio', 'have_face_cate'
photo_features = ['exposure_num'] + face_features + text_features
user_features = user_action_features + user_face_favor_features + user_text_favor_features

y_label = ['click']

features_to_train = user_features + photo_features + time_features
# features_to_train = list(set(features_to_train) - set(['clicked_ratio']))

norm_features = ['browse_num', 'click_num', 'like_num', 'follow_num', 'playing_sum', 'duration_sum', 'click_ratio',
                 'like_ratio', 'follow_ratio', 'playing_ratio', 'browse_time_diff', 'click_freq', 'browse_freq',
                 'playing_freq', 'man_favor', 'woman_favor', 'man_cv_favor', 'woman_cv_favor', 'man_age_favor',
                 'woman_age_favor', 'man_yen_value_favor', 'woman_yen_value_favor', 'face_click_favor',
                 'non_face_click_favor', 'cover_length_favor', 'exposure_num', 'face_num', 'man_num', 'woman_num',
                 'man_scale', 'woman_scale', 'human_scale', 'man_avg_age', 'woman_avg_age', 'human_avg_age',
                 'man_avg_attr', 'woman_avg_attr', 'human_avg_attr', 'cover_length', 'avg_tfidf', 'key_words_num',
                 'time', 'duration_time', 'period_click_ratio']

uint64_cols = ['user_id', 'photo_id', 'time']
uint32_cols = ['playing_sum', 'browse_time_diff', 'duration_sum']
uint16_cols = ['browse_num', 'exposure_num', 'click_num', 'duration_time', 'like_num', 'follow_num', 'clicked_num']
uint8_cols = ['cover_length', 'man_num', 'woman_num', 'face_num', 'time_cate', 'duration_time_cate']
bool_cols = ['have_face_cate', 'click']
float32_cols = ['period_click_ratio', 'clicked_ratio','non_face_click_favor', 'face_click_favor', 'man_favor', 'woman_avg_age', 'playing_freq', 'woman_age_favor', 'woman_yen_value_favor', 'human_scale', 'woman_favor', 'click_freq', 'woman_cv_favor', 'man_age_favor', 'man_yen_value_favor', 'follow_ratio', 'man_scale', 'browse_freq', 'man_avg_age', 'man_cv_favor', 'man_avg_attr', 'playing_ratio', 'woman_scale', 'click_ratio', 'human_avg_age', 'woman_avg_attr', 'like_ratio', 'cover_length_favor', 'human_avg_attr', 'avg_tfidf', 'hour_click_ratio']
float64_cols = []


feature_dtype_map = {}
for name in uint64_cols:
    feature_dtype_map.update({name: 'uint64'})
for name in uint32_cols:
    feature_dtype_map.update({name: 'uint32'})
for name in uint16_cols:
    feature_dtype_map.update({name: 'uint16'})
for name in uint8_cols:
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
