from konlpy.tag import Mecab
from tqdm import tqdm
import re
import pickle
import csv
import pandas as pd
from pandas import DataFrame
import numpy as np
import gensim
import matplotlib.pyplot as plt
import nltk
import pyLDAvis.gensim_models
import regex
import seaborn as sns
import streamlit as st
from gensim import corpora
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess

import streamlit.components.v1 as components
import matplotlib.colors as mcolors
import plotly.express as px

DEFAULT_HIGHLIGHT_PROBABILITY_MINIMUM = 0.001
DEFAULT_NUM_TOPICS = 4

DATASETS = {
    '소셜벤처데이터': {
        'path': './data/소셜벤처실태조사_qual.csv',
        'url': 'https://www.kbiz.or.kr/ko/contest/view.do?seq=41&mnSeq=1202',
        'description': (
            'Data source: KBIZ 중소기업 중앙회, 중소기업 통계데이터' 
        )
    }
}


def lda_options():
    return {
        'num_topics': st.number_input('Number of Topics', min_value=1, value=9,
                                      help='The number of requested latent topics to be extracted from the training corpus.'),
        'chunksize': st.number_input('Chunk Size', min_value=1, value=2000,
                                     help='Number of documents to be used in each training chunk.'),
        'passes': st.number_input('Passes', min_value=1, value=1,
                                  help='Number of passes through the corpus during training.'),
        'update_every': st.number_input('Update Every', min_value=1, value=1,
                                        help='Number of documents to be iterated through for each update. Set to 0 for batch learning, > 1 for online iterative learning.'),
        'alpha': st.selectbox('𝛼', ('symmetric', 'asymmetric', 'auto'),
                              help='A priori belief on document-topic distribution.'),
        'eta': st.selectbox('𝜂', (None, 'symmetric', 'auto'), help='A-priori belief on topic-word distribution'),
        'decay': st.number_input('𝜅', min_value=0.5, max_value=1.0, value=0.5,
                                 help='A number between (0.5, 1] to weight what percentage of the previous lambda value is forgotten when each new document is examined.'),
        'offset': st.number_input('𝜏_0', value=1.0,
                                  help='Hyper-parameter that controls how much we will slow down the first steps the first few iterations.'),
        'eval_every': st.number_input('Evaluate Every', min_value=1, value=10,
                                      help='Log perplexity is estimated every that many updates.'),
        'iterations': st.number_input('Iterations', min_value=1, value=50,
                                      help='Maximum number of iterations through the corpus when inferring the topic distribution of a corpus.'),
        'gamma_threshold': st.number_input('𝛾', min_value=0.0, value=0.001,
                                           help='Minimum change in the value of the gamma parameters to continue iterating.'),
        'minimum_probability': st.number_input('Minimum Probability', min_value=0.0, max_value=1.0, value=0.01,
                                               help='Topics with a probability lower than this threshold will be filtered out.'),
        'minimum_phi_value': st.number_input('𝜑', min_value=0.0, value=0.01,
                                             help='if per_word_topics is True, this represents a lower bound on the term probabilities.'),
        'per_word_topics': st.checkbox('Per Word Topics',
                                       help='If True, the model also computes a list of topics, sorted in descending order of most likely topics for each word, along with their phi values multiplied by the feature length (i.e. word count).')
    }


def nmf_options():
    return {
        'num_topics': st.number_input('Number of Topics', min_value=1, value=9, help='Number of topics to extract.'),
        'chunksize': st.number_input('Chunk Size', min_value=1, value=2000,
                                     help='Number of documents to be used in each training chunk.'),
        'passes': st.number_input('Passes', min_value=1, value=1,
                                  help='Number of full passes over the training corpus.'),
        'kappa': st.number_input('𝜅', min_value=0.0, value=1.0, help='Gradient descent step size.'),
        'minimum_probability': st.number_input('Minimum Probability', min_value=0.0, max_value=1.0, value=0.01,
                                               help='If normalize is True, topics with smaller probabilities are filtered out. If normalize is False, topics with smaller factors are filtered out. If set to None, a value of 1e-8 is used to prevent 0s.'),
        'w_max_iter': st.number_input('W max iter', min_value=1, value=200,
                                      help='Maximum number of iterations to train W per each batch.'),
        'w_stop_condition': st.number_input('W stop cond', min_value=0.0, value=0.0001,
                                            help=' If error difference gets less than that, training of W stops for the current batch.'),
        'h_max_iter': st.number_input('H max iter', min_value=1, value=50,
                                      help='Maximum number of iterations to train h per each batch.'),
        'h_stop_condition': st.number_input('W stop cond', min_value=0.0, value=0.001,
                                            help='If error difference gets less than that, training of h stops for the current batch.'),
        'eval_every': st.number_input('Evaluate Every', min_value=1, value=10,
                                      help='Number of batches after which l2 norm of (v - Wh) is computed.'),
        'normalize': st.selectbox('Normalize', (True, False, None), help='Whether to normalize the result.')
    }
