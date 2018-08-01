
categorical_features = ['gender', 'photo_cluster_label', 'text_cluster_label', "have_text_cate", "have_face_cate", "text_class_label"]

to_discretization_features = ['time', 'duration_time'] + ["key_words_num", "exposure_num", "browse_num", "face_num", "woman_num", "man_num",
                                     "age", "appearance", "cover_length"]



float32_cols = ['cover_words_ctr', 'face_num_ctr', 'woman_num_ctr', 'man_num_ctr', 'gender_ctr',
                    'age_ctr', 'appearance_ctr', 'cover_length_ctr', 'duration_time_ctr', 'time_ctr', 'photo_cluster_label_ctr'] + \
               ['non_face_click_favor', 'face_click_favor',
                                                              'man_favor', 'woman_avg_age', 'woman_age_favor',
                                                              'woman_yen_value_favor',
                                                              'human_scale', 'woman_favor', 'woman_cv_favor',
                                                              'man_age_favor', 'man_yen_value_favor',
                                                              'follow_ratio', 'man_scale', 'man_avg_age',
                                                              'man_cv_favor', 'man_avg_attr',
                                                              'playing_ratio', 'woman_scale', 'click_ratio',
                                                              'human_avg_age', 'woman_avg_attr', 'like_ratio',
                                                              'cover_length_favor', 'human_avg_attr', 'avg_tfidf',
                                                              'woman_num_ratio',
                                                              'man_num_ratio', 'duration_favor',
                                                              'face_favor', 'text_clicked_ratio', 'scale']

cate_features = categorical_features + to_discretization_features
numeric_features = float32_cols

real_value_vector_features = ['cover_words', 'visual', 'pos_photo_id', 'neg_photo_id', 'pos_photo_cluster_label',
                              'neg_photo_cluster_label', 'pos_user_id', 'neg_user_id']

features_to_train = cate_features + numeric_features

ignore_features = []

y_label = ['click']

seed = 2018

