#coding:utf8    
    
# params config

# model config

# feature config
user_action_features = ['browse_num', 'click_num', 'like_num', 'follow_num', 'playing_sum','duration_sum', 'click_ratio', 'like_ratio', 'follow_ratio', 'playing_ratio', 'browse_time_diff', 'click_freq', 'browse_freq', 'playing_freq']
    
face_features = ['face_num', 'man_num', 'woman_num', 'man_scale', 'woman_scale', 'human_scale', 'man_avg_age', 'woman_avg_age', 'human_avg_age',  'man_avg_attr', 'woman_avg_attr', 'human_avg_attr']
    
    
user_face_favor_features = ['man_favor', 'woman_favor', 'man_cv_favor', 'woman_cv_favor', 'man_age_favor', 'woman_age_favor', 'man_yen_value_favor', 'woman_yen_value_favor', 'face_click_favor', 'non_face_click_favor']

user_text_favor_features = ['cover_length_favor']
    
    
id_features = ['user_id', 'photo_id']
time_features = ['time', 'duration_time']

text_features = ['cover_length', 'avg_tfidf']

#'clicked_ratio'
photo_features = ['exposure_num', 'have_face_cate', 'clicked_ratio'] + face_features + text_features
user_features = user_action_features + user_face_favor_features + user_text_favor_features

y_label = ['click']

features_to_train = user_features + photo_features + time_features
# features_to_train = list(set(features_to_train) - set(['clicked_ratio']))

# BayesianSmoothing parameters, this already trained
alpha = 2.5171267342473382
beta = 7.087836849232511
