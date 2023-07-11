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
        'path': '/data/소셜벤처실태조사_qual.csv',
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
    

MODELS = {
    'Latent Dirichlet Allocation': {
        'options': lda_options,
        'class': gensim.models.LdaModel,
        'help': 'https://radimrehurek.com/gensim/models/ldamodel.html'
    }
}

COLORS = [color for color in mcolors.XKCD_COLORS.values()]

@st.experimental_memo()
def generate_texts_df(selected_dataset: str):
    dataset = DATASETS[selected_dataset]
    return pd.read_csv(f'{dataset["path"]}')


@st.experimental_memo()
def denoise_docs(texts_df: pd.DataFrame, text_column: str):
    texts = texts_df[text_column].values.tolist()
    docs = [[w for w in simple_preprocess(doc, deacc=True) if w not in stopwords.words('english')] for doc in texts]
    return docs

@st.experimental_memo()
def prepare_training_data(docs):
    id2word = corpora.Dictionary(docs)
    corpus = [id2word.doc2bow(doc) for doc in docs]
    return id2word, corpus


@st.experimental_memo()
def train_model(docs, base_model, **kwargs):
    id2word, corpus = prepare_training_data(docs)
    model = base_model(corpus=corpus, id2word=id2word, **kwargs)
    return id2word, corpus, model


def clear_session_state():
    for key in ('model_kwargs', 'id2word', 'corpus', 'model', 'previous_perplexity', 'previous_coherence_model_value'):
        if key in st.session_state:
            del st.session_state[key]


def calculate_perplexity(model, corpus):
    return np.exp2(-model.log_perplexity(corpus))


def calculate_coherence(model, corpus, coherence):
    coherence_model = CoherenceModel(model=model, corpus=corpus, coherence=coherence)
    return coherence_model.get_coherence()


@st.experimental_memo()
def white_or_black_text(background_color):
    # https://stackoverflow.com/questions/3942878/how-to-decide-font-color-in-white-or-black-depending-on-background-color
    red = int(background_color[1:3], 16)
    green = int(background_color[3:5], 16)
    blue = int(background_color[5:], 16)
    return 'black' if (red * 0.299 + green * 0.587 + blue * 0.114) > 186 else 'white'


def perplexity_section():
    with st.spinner('Calculating Perplexity ...'):
        perplexity = calculate_perplexity(st.session_state.model, st.session_state.corpus)
    key = 'previous_perplexity'
    delta = f'{perplexity - st.session_state[key]:.4}' if key in st.session_state else None
    st.metric(label='Perplexity', value=f'{perplexity:.4f}', delta=delta, delta_color='inverse')
    st.session_state[key] = perplexity
    st.markdown('Viz., https://en.wikipedia.org/wiki/Perplexity')
    st.latex(r'Perplexity = \exp\left(-\frac{\sum_d \log(p(w_d|\Phi, \alpha))}{N}\right)')

