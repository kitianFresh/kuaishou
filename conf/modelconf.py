#coding:utf8
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
xgboost = True
lgbm = True
try:
    from xgboost import XGBClassifier
except ImportError as e:
    print(e)
    xgboost = False
try:
    from lightgbm import LGBMClassifier
except ImportError as e:
    print(e)
    lgbm = False
    
    
    
# params config

# model config

models = {
    "LightGBM": LGBMClassifier() if lgbm else None,
    "LogisticRegression": LogisticRegression(C=1),
    "LinearSVM": svm.SVC(kernel="linear", C=0.025),
    "RBFSVM": svm.SVC(gamma=0.01, C=10),
    "DecisionTree": DecisionTreeClassifier(max_depth=5),
    "RandomForest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    "AdaBoost": AdaBoostClassifier(),
    "GBDT": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, min_samples_leaf=9),
    "XGBoost": XGBClassifier() if xgboost else None,
}

classifiers = {
    name: clf for name, clf in models.iteritems() if clf is not None
}

# feature config
user_action_features = ['browse_num', 'click_num', 'like_num', 'follow_num', 'playing_sum','duration_sum', 'click_ratio', 'like_ratio', 'follow_ratio', 'playing_ratio', 'browse_time_diff', 'click_freq', 'browse_freq', 'playing_freq']
    
face_features = ['face_num', 'man_num', 'woman_num', 'man_scale', 'woman_scale', 'human_scale', 'man_avg_age', 'woman_avg_age', 'human_avg_age',  'man_avg_attr', 'woman_avg_attr', 'human_avg_attr']
    
    
user_face_favor_features = ['face_favor', 'man_favor', 'woman_favor', 'man_cv_favor', 'woman_cv_favor', 'man_age_favor', 'woman_age_favor', 'man_yen_value_favor', 'woman_yen_value_favor']

user_text_favor_features = ['cover_length_favor']
    
    
id_features = ['user_id', 'photo_id']
time_features = ['time', 'duration_time']

text_features = ['cover_length']

photo_features = ['exposure_num', 'have_face_cate'] + face_features + text_features
user_features = user_action_features + user_face_favor_features + user_text_favor_features

y_label = ['click']

features_to_train = user_features + photo_features + time_features

# BayesianSmoothing parameters, this already trained
alpha = 2.5171267342473382
beta = 7.087836849232511
