import os
import numpy as np

from label import run, parser
from label.lfs import FramingLabels
from label.lfs.framing import get_lfs


def load_embeddings(filename):
    return np.load(filename)


REGISTERED_MODEL_NAME = 'FramingLabelModel'
LF_FEATURES = {
    # 'txt_clean_roberta': load_embeddings,
    # 'txt_clean_use': load_embeddings,
    "txt_clean_mpnet": load_embeddings,
    "txt_clean_climate": load_embeddings
    }
DEV_ANNOTATIONS_PATH = os.path.join('/annotations', 'framing', 'gold_df.pkl')


def main():
    parser.add_argument('--trld', default=0.5, type=float, help='cosine similarity threshold')
    # parser.add_argument('--encoder', default='roberta', choices=('roberta', 'use'), type=str,
    #                     help='which encoder embeddings to use')
    parser.add_argument('--encoder', default='mpnet', choices=('mpnet', 'climate'), type=str,
                        help='which encoder embeddings to use')
    parsed_args = parser.parse_args()
    run.start(REGISTERED_MODEL_NAME, LF_FEATURES, DEV_ANNOTATIONS_PATH, get_lfs, FramingLabels, parsed_args)


if __name__ == '__main__':
    main()
