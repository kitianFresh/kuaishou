#coding:utf8


import unittest
import sys
sys.path.append('..')
from common.base import Classifier

class TestMarkdownPy(unittest.TestCase):
    def setUp(self):
        path = '../features'
        USE_SAMPLE = True
        fmt = 'csv'
        version = '1.1.1'
        desc = "测试"

        model_name = 'lr'

        feature_store_path = '../sample/features' if USE_SAMPLE else '../data/features'

        model_store_path = './sample/' if USE_SAMPLE else './data'

        col_feature_store_path = '../sample/features/columns' if USE_SAMPLE else '../data/features/columns'

        self.model = Classifier(clf=None, dir=model_store_path,
                           name=model_name, version=version,
                           description=desc, features_to_train=[])

    def test_save(self):
        '''
        Non-marked lines should only get 'p' tags around all input
        '''
        # self.assertEqual(
        #     run_markdown('this line has no special handling'),
        #     'this line has no special handling</p>')
        self.model.save()

if __name__ == '__main__':
    unittest.main()