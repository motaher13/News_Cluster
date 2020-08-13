import os
import logging
import tarfile
import pickle
import shutil
import re
import codecs
import json
import numpy as np
import scipy.sparse as sp
import random

from base import get_data_home
from base import load_files
from base import _pkl_filepath
from base import _fetch_remote
from base import RemoteFileMetadata
from sklearn.utils import check_random_state,Bunch

logger = logging.getLogger(__name__)


CACHE_NAME = "processed_data.pkz"


def convert_to_pkz(target_dir, cache_path):

    cache = dict(data=load_files(target_dir, encoding='UTF-8'))
    compressed_content = codecs.encode(pickle.dumps(cache), 'zlib_codec')
    with open(cache_path, 'wb') as f:
        f.write(compressed_content)

    #shutil.rmtree(target_dir)
    return cache



# def download_20newsgroups(target_dir, cache_path):
    # """Download the 20 newsgroups data and stored it as a zipped pickle."""

    # with open('data.json', 'r',encoding='utf-8') as f:
        # data = json.load(f)
        # data=data['response']['docs']
    
    # random.shuffle(data)
    # train=list()
    # test=list()
    # for i in range(0,70000):
        # st = data[i]['content']
        # st=re.sub("\n","",st)
        # st=re.sub("\\\.*?\s","",st)
        # train.append(st)
    # for i in range(70000,99990):
        # st = data[i]['content']
        # st=re.sub("\n","",st)
        # st=re.sub("\\\.*?\s","",st)
        # test.append(st)
    # train=dict(data=train)
    # test=dict(data=test)
    # cache = dict(train=train,
                 # test=test)
    # print(cache)
    # compressed_content = codecs.encode(pickle.dumps(cache), 'zlib_codec')
    # with open(cache_path, 'wb') as f:
        # f.write(compressed_content)

    # #shutil.rmtree(target_dir)
    # return cache




def fetch_data_groups(data_home=None, subset='train', categories=None,
                       shuffle=False, random_state=42,
                       remove=(),
                       process_if_missing=True):
    """Load the filenames and data from dataset.

    Read more in the :ref:`User Guide <20newsgroups>`.

    Parameters
    ----------
    data_home : optional, default: None
        Specify a download and cache folder for the datasets. If None,
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    subset : 'train' or 'test', 'all', optional
        Select the dataset to load: 'train' for the training set, 'test'
        for the test set, 'all' for both, with shuffled ordering.

    categories : None or collection of string or unicode
        If None (default), load all the categories.
        If not None, list of category names to load (other categories
        ignored).

    shuffle : bool, optional
        Whether or not to shuffle the data: might be important for models that
        make the assumption that the samples are independent and identically
        distributed (i.i.d.), such as stochastic gradient descent.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset shuffling. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    remove : tuple
        May contain any subset of ('headers', 'footers', 'quotes'). Each of
        these are kinds of text that will be detected and removed from the
        newsgroup posts, preventing classifiers from overfitting on
        metadata.

        'headers' removes newsgroup headers, 'footers' removes blocks at the
        ends of posts that look like signatures, and 'quotes' removes lines
        that appear to be quoting another post.

        'headers' follows an exact standard; the other filters are not always
        correct.

    download_if_missing : optional, True by default
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source site.
    """

    data_home = get_data_home(data_home=data_home)
    cache_path = _pkl_filepath(data_home, CACHE_NAME)
    #twenty_home = os.path.join(data_home, "20news_home")
    twenty_home = data_home
    cache = None
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                compressed_content = f.read()
            uncompressed_content = codecs.decode(
                compressed_content, 'zlib_codec')
            cache = pickle.loads(uncompressed_content)
        except Exception as e:
            print(80 * '_')
            print('Cache loading failed')
            print(80 * '_')
            print(e)

    if cache is None:
        if process_if_missing:
            logger.info("Processing data. "
                        "This may take a few minutes.")
            cache = convert_to_pkz(target_dir=twenty_home,
                                          cache_path=cache_path)
        else:
            raise IOError('dataset not found')

    data = cache['data']

    data['description'] = 'given dataset'


    if shuffle:
        random_state = check_random_state(random_state)
        indices = np.arange(data.target.shape[0])
        random_state.shuffle(indices)
        data.filenames = data.filenames[indices]
        data.target = data.target[indices]
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data.data, dtype=object)
        data_lst = data_lst[indices]
        data.data = data_lst.tolist()

    return data

