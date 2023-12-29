import importlib
import logging
import os
import sys

import requests
from tqdm import tqdm


def auto_import(name, is_class=False):
    """ Import from the external python packages.
    """
    def __get_module(comps_list):
        return importlib.import_module(".".join(comps_list))

    components = name.split('.')
    m = getattr(__get_module(components[:-1]), components[-1])

    return m() if is_class else m


def setup_custom_logger(name, add_screen_handler=False, filepath=None):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if add_screen_handler:
        screen_handler = logging.StreamHandler(stream=sys.stdout)
        screen_handler.setFormatter(formatter)
        logger.addHandler(screen_handler)

    if filepath is not None:
        handler = logging.FileHandler(filepath, mode='w')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def download(dest_file_path, source_url):
    """ Refered to https://github.com/nicolay-r/ner-bilstm-crf-tensorflow/blob/master/ner/utils.py
        Simple http file downloader
    """
    print(('Downloading from {src} to {dest}'.format(src=source_url, dest=dest_file_path)))

    sys.stdout.flush()
    datapath = os.path.dirname(dest_file_path)

    if not os.path.exists(datapath):
        os.makedirs(datapath, mode=0o755)

    dest_file_path = os.path.abspath(dest_file_path)

    r = requests.get(source_url, stream=True)
    total_length = int(r.headers.get('content-length', 0))

    with open(dest_file_path, 'wb') as f:
        pbar = tqdm(total=total_length, unit='B', unit_scale=True)
        for chunk in r.iter_content(chunk_size=32 * 1024):
            if chunk:  # filter out keep-alive new chunks
                pbar.update(len(chunk))
                f.write(chunk)


def get_default_download_dir():
    """ Refered to NLTK toolkit approach
        https://github.com/nltk/nltk/blob/8e771679cee1b4a9540633cc3ea17f4421ffd6c0/nltk/downloader.py#L1051
    """

    # On Windows, use %APPDATA%
    if sys.platform == "win32" and "APPDATA" in os.environ:
        homedir = os.environ["APPDATA"]

    # Otherwise, install in the user's home directory.
    else:
        homedir = os.path.expanduser("~/")
        if homedir == "~/":
            raise ValueError("Could not find a default download directory")

    return os.path.join(homedir, ".arekit")
