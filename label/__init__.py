import argparse
import os
import sys
import logging


parser = argparse.ArgumentParser(description='Perform data programming.')
parser.add_argument('--step', default=0, help='create label matrix (1), label model (2), training data (3). or all (0)')
parser.add_argument('--eval', default=0, help='evaluate labeling functions (1), label model (2), or both (0)')
parsed_args = parser.parse_args()

TRAIN_CSV_FILENAME = os.path.join('label', 'data', 'unlabeled_train.csv')
TRAIN_DF_FILENAME = os.path.join('label', 'resources', 'train_df.pkl')
TRAIN_MATRIX_FILENAME = os.path.join('label', 'resources', 'train_matrix.npy')
LABEL_MODEL_FILENAME = os.path.join('label', 'resources', 'label_model.pkl')
TRAINING_DATA_FILENAME = os.path.join('label', 'data', 'training_data.pkl')

DEMO_CSV_FILENAME = os.path.join('demo', 'demo_data.csv')
DEMO_DF_FILENAME = os.path.join('demo', 'demo_df.pkl')
DEMO_MATRIX_FILENAME = os.path.join('demo', 'demo_matrix.npy')
DEMO_LABEL_MODEL_FILENAME = os.path.join('demo', 'demo_label_model.pkl')
DEMO_TRAINING_DATA_FILENAME = os.path.join('demo', 'demo_training_data.pkl')

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
