"""
https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import util


def plot_similarity(labels, features, figsize, rotation, title):
    cos_sim = util.cos_sim(features, features)
    sns.set(font_scale=1.2, rc={'figure.figsize': (figsize, figsize)})
    g = sns.heatmap(
        cos_sim,
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        cmap="YlOrRd")
    g.set_xticklabels(labels, rotation=rotation)
    g.set_title(title)

    plt.tight_layout()
    plt.savefig('/figures/{}.jpeg'.format(title))
    plt.clf()


def run_and_plot():
    df = pd.read_sql("SELECT * FROM frame_elements;", 'crate://crate-db:4200').sort_values(by=['frame'])
    df['mpnet'] = df['element_mpnet'].apply(lambda x: np.load(x))
    df['bertweet'] = df['element_bertweet'].apply(lambda x: np.load(x))
    mpnet_embedding_matrix = np.array(df['mpnet'].tolist())
    bertweet_embedding_matrix = np.array(df['bertweet'].tolist())
    frame_elements = df['element_txt'].tolist()
    plot_similarity(frame_elements, mpnet_embedding_matrix, 100, 90, 'MPNET-STS')
    plot_similarity(frame_elements, bertweet_embedding_matrix, 100, 90, 'BERTWEET-STS')

    for frame in df.frame.unique():
        frame_df = df.loc[(df.frame == frame)]
        mpnet_embedding_matrix = np.array(frame_df['mpnet'].tolist())
        bertweet_embedding_matrix = np.array(frame_df['bertweet'].tolist())
        frame_elements = frame_df['element_txt'].tolist()
        plot_similarity(frame_elements, mpnet_embedding_matrix, 13, 90, "{}-MPNET-STS".format(frame))
        plot_similarity(frame_elements, bertweet_embedding_matrix, 13, 90, "{}-BERTWEET-STS".format(frame))


if __name__ == '__main__':
    run_and_plot()
