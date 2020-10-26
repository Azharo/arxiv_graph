import time
import re
import sys
import os
import gzip
import math
import json
from pathlib import Path
import config
import src.blob_data_transfer as blob_pull
from src.regex_arxiv import REGEX_ARXIV_FLEXIBLE, clean

RE_FLEX = re.compile(REGEX_ARXIV_FLEXIBLE)
RE_OLDNAME_SPLIT = re.compile(r"([a-z\-]+)(\d+)")

def path_to_id(blob):
    """
    Convert filepath name of ArXiv file to ArXiv ID.
    Need to remove the ".txt" from file names first if they have it
    """
    name = os.path.splitext(os.path.basename(blob))[0]
    name.replace('.txt','')
    if '.' in name:  # new  ID
        return name 
    split = [a for a in RE_OLDNAME_SPLIT.split(name) if a]
    return "/".join(split)

def get_text_stream(blob):
    return blob_pull.stream_blob(blob).decode()

def extract_references(txt, pattern=RE_FLEX):
    """
    Parameters
    ----------
        filename : str
            name of file to search for pattern
        pattern : re pattern object
            compiled regex pattern

    Returns
    -------
        citations : list
            list of found arXiv IDs
    """
    out = []
    for matches in pattern.findall(txt):
        out.extend([clean(a) for a in matches if a])
    return list(set(out))

def citation_list_inner(article):
    """ Find references in all the input articles
    Parameters
    ----------
        article : str
            path to article blob
    Returns
    -------
        citations : dict[arXiv ID] = list of arXiv IDs
            dictionary of articles and their references
    """
    cites = {}
    try:
        article_text = get_text_stream(article)
        refs = extract_references(article_text)
        cites[path_to_id(article)] = refs
        return cites
    except Exception as e:
        print("Error in {}".format(article))
        print(e)
        #log.error("Error in {}".format(article))
        
def default_filename():
    return os.path.join(os.getcwd(), 'internal-citations.json.gz')

def save_to_default_location(citations):
    filename = default_filename()

    with gzip.open(filename, 'a+') as fn:
        json_data = json.dumps(citations).encode('utf-8')
        fn.write(json_data + '\n'.encode('utf-8'))