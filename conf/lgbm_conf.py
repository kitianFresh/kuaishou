# params config

# model config

# feature config
# 'hour_click_ratio', 'period_click_ratio', 'click_ratio'， 'playing_duration_favor'
user_action_features = ['click_ratio', 'browse_num', 'click_num', 'like_num', 'follow_num', 'playing_sum',
                        'duration_sum', 'like_ratio', 'follow_ratio', 'playing_ratio', 'browse_time_diff', 'click_freq',
                        'browse_freq', 'playing_freq', 'duration_favor']

face_features = ['face_num', 'man_num', 'woman_num', 'man_scale', 'woman_scale', 'human_scale', 'man_avg_age',
                 'woman_avg_age',
                 'human_avg_age', 'man_avg_attr', 'woman_avg_attr', 'human_avg_attr', 'woman_num_ratio',
                 'man_num_ratio']
photo_max_face_features = ['scale', 'gender', 'age', 'appearance']

user_face_favor_features = ['face_favor', 'man_favor', 'woman_favor', 'man_cv_favor', 'woman_cv_favor', 'man_age_favor',
                            'woman_age_favor', 'man_yen_value_favor', 'woman_yen_value_favor', 'face_click_favor',
                            'non_face_click_favor']

user_text_favor_features = ['cover_length_favor']

id_features = ['user_id', 'photo_id']
expand_id_features = ['uid0', 'uid1', 'uid2', 'uid3', 'uid4', 'uid5', 'uid6', 'uid7', 'uid8', 'uid9', 'uid10', 'uid11',
                      'uid12', 'uid13', 'uid14', 'uid15']
time_features = ['time', 'duration_time']
# time_features = ['time_cate', 'duration_time_cate']

# 0.0006 max_word_ctr
text_features = ['cover_length', 'avg_tfidf', 'key_words_num', 'text_class_label', 'text_cluster_label',
                 'text_cluster_exposure_num', 'text_clicked_ratio']
# text_features = ['cover_length', 'avg_tfidf', 'key_words_num','text_class_label','text_cluster_label']
# text_features = ['cover_length', 'avg_tfidf', 'key_words_num']

# visual_features = ['photo_cluster_label', 'photo_class_label']
visual_features = ['photo_cluster_label']

# 'clicked_ratio', 'have_face_cate'
photo_features = ['exposure_num', 'have_face_cate', 'have_text_cate'] + face_features + text_features + visual_features
user_features = user_action_features + user_face_favor_features + user_text_favor_features

# combine_ctr_features = ['user_id_face_num_ctr', 'user_id_woman_num_ctr', 'user_id_man_num_ctr', 'user_id_gender_ctr', 'user_id_age_ctr', 'user_id_appearance_ctr', 'user_id_cover_length_ctr', 'user_id_duration_time_ctr']
combine_ctr_features = []
one_ctr_features = ['max_word_ctr', 'face_num_ctr', 'woman_num_ctr', 'man_num_ctr', 'gender_ctr', 'age_ctr',
                    'appearance_ctr', 'cover_length_ctr', 'duration_time_ctr', 'time_ctr']

y_label = ['click']

features_to_train = user_features + photo_features + time_features + combine_ctr_features + one_ctr_features
# features_to_train = list(set(features_to_train) - set(['clicked_ratio']))