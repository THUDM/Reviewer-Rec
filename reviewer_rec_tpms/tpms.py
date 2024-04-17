# This Python file uses the following encoding: utf-8
import re
import unicodedata
import argparse
import os
import json
import math

from nltk.stem import PorterStemmer
from collections import Counter
from collections import defaultdict

import numpy as np
import json
import random
import os
import time


###############################################################
# Helper file for TPMS
# The code is adapted from the original code by Laurent Charlin
###############################################################

def isUIWord(word):
    """
        Returns true if the word is un-informative
        (for now either a stopword or a single character word)
    """

    if len(word) <= 1:
        return True

    # List provided by Amit Gruber
    l = set(['a', 'about', 'above', 'accordingly', 'across', 'after', 'afterwards', 'again', \
             'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', \
             'always', 'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', \
             'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', \
             'around', 'as', 'aside', 'at', 'away', 'back', 'be', 'became', 'because', \
             'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', \
             'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', \
             'bottom', 'briefly', 'but', 'by', 'call', 'came', 'can', 'cannot', 'cant', \
             'certain', 'certainly', 'co', 'computer', 'con', 'could', 'couldnt', 'cry', \
             'de', 'describe', 'detail', 'do', 'does', 'done', 'down', 'due', 'during', \
             'each', 'edit', 'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', \
             'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', \
             'everywhere', 'except', 'few', 'fifteen', 'fify', 'fill', 'find', 'fire', \
             'first', 'five', 'following', 'for', 'former', 'formerly', 'forty', 'found', \
             'four', 'from', 'front', 'full', 'further', 'gave', 'get', 'gets', 'give', \
             'given', 'giving', 'go', 'gone', 'got', 'had', 'hardly', 'has', 'hasnt', 'have', \
             'having', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', \
             'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', \
             'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', \
             'it', 'its', 'itself', 'just', 'keep', 'kept', 'kg', 'knowledge', 'largely', \
             'last', 'latter', 'latterly', 'least', 'less', 'like', 'ltd', 'made', 'mainly', \
             'make', 'many', 'may', 'me', 'meanwhile', 'mg', 'might', 'mill', 'mine', 'ml', \
             'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my', 'myself', \
             'name', 'namely', 'nearly', 'necessarily', 'neither', 'never', 'nevertheless', \
             'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'normally', 'not', \
             'noted', 'nothing', 'now', 'nowhere', 'obtain', 'obtained', 'of', 'off', \
             'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', \
             'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'owing', 'own', 'part', \
             'particularly', 'past', 'per', 'perhaps', 'please', 'poorly', 'possible', \
             'possibly', 'potentially', 'predominantly', 'present', 'previously', \
             'primarily', 'probably', 'prompt', 'promptly', 'put', 'quickly', 'quite', \
             'rather', 're', 'readily', 'really', 'recently', 'refs', 'regarding', \
             'regardless', 'relatively', 'respectively', 'resulted', 'resulting', 'results', 'rst', \
             'said', 'same', 'second', 'see', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'serious', \
             'several', 'shall', 'she', 'should', 'show', 'showed', 'shown', 'shows', 'side', \
             'significantly', 'similar', 'similarly', 'since', 'sincere', 'six', 'sixty', \
             'slightly', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', \
             'sometimes', 'somewhat', 'somewhere', 'soon', 'specifically', 'state', 'states', \
             'still', 'strongly', 'substantially', 'successfully', 'such', 'sufficiently', \
             'system', 'take', 'ten', 'than', 'that', 'the', 'their', 'theirs', 'them', \
             'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', \
             'therein', 'thereupon', 'these', 'they', 'thick', 'thin', 'third', 'this', \
             'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', \
             'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', \
             'under', 'unless', 'until', 'up', 'upon', 'us', 'use', 'used', 'usefully', \
             'usefulness', 'using', 'usually', 'various', 'very', 'via', 'was', 'we', 'well', \
             'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', \
             'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', \
             'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'widely', \
             'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', \
             'yourself', 'yourselves'])

    if word in l:
        return True

        # if re.match('[0-9]*$', word):
    # return True

    return False


def tokenize(line):
    # space_regexp = re.compile('\s', re.U)
    # space_regexp_full = re.compile('\W', re.U)

    space_regexp = re.compile('[^a-zA-Z]')  # üñèéóãàáíøö¨úïäýâåìçôêßëîÁÅÇÉÑÖØÜ]')
    line = sanitize(line)  # sanitize returns unicode
    words = re.split(space_regexp, line)
    words = [x for x in words if len(x) > 0]

    return words


def sanitize(w):
    """
      sanitize (remove accents and standardizes)
    """

    # print w

    map = {'æ': 'ae',
           'ø': 'o',
           '¨': 'o',
           'ß': 'ss',
           'Ø': 'o',
           '\xef\xac\x80': 'ff',
           '\xef\xac\x81': 'fi',
           '\xef\xac\x82': 'fl'}

    # This replaces funny chars in map
    for char, replace_char in map.items():
        w = re.sub(char, replace_char, w)

    # w = unicode(w, encoding='latin-1')
    # w = str(w, encoding="utf-8")

    # This gets rite of accents
    w = ''.join((c for c in unicodedata.normalize('NFD', w) if unicodedata.category(c) != 'Mn'))

    return w


def _wrap_na(text):
    """Substitute None with empty string.

    Args:
        text: Text to be wrapped

    Returns:
        Empty string if text is none and the original text otherwise
    """

    if text is None:
        return ''

    return text


def paper2bow(text):
    """Tokenize and filter given text.

    Args:
        text: string to be processed
    Returns:
        Counter of words in the text
    """
    words = [w.lower() for w in tokenize(text)]
    # Filter out uninformative words.
    words = filter(lambda w: not isUIWord(w), words)
    # Use PortStemmer.
    ps = PorterStemmer()
    words = [ps.stem(w) for w in words]
    return Counter(words)


def get_paper_profiles(papers, regime):

    papers_dict = {}
    # ps = PorterStemmer()

    for key in papers.keys():
        paper = papers[key]
        texts = []
        loc_counter = {}

        texts.append(_wrap_na(paper['Title']))

        content = paper2bow(' '.join(texts))

        for word in loc_counter:
            if word in content:
                content[word] += loc_counter[word]
            else:
                content[word] = loc_counter[word]

        papers_dict[key] = content

    return papers_dict


def get_reviewer_profiles(reviewers, papers, inter, regime):

    profiles_dict = {}
    # ps = PorterStemmer()

    for key in reviewers.keys():
        texts = []
        reviewer = reviewers[key]
        loc_counter = {}
        if key not in inter.keys():
            continue
        for rev_paper_id in inter[reviewer['u_newid']]:
            rev_paper = papers[rev_paper_id]
            texts.append(_wrap_na(rev_paper['Title']))

        profile = paper2bow(' '.join(texts))

        for word in loc_counter:
            if word in profile:
                profile[word] += loc_counter[word]
            else:
                profile[word] = loc_counter[word]

        profiles_dict[key] = profile

    return profiles_dict


def compute_idf(profiles):
    """Given a set of documents, compute IDF of each word

    Args:
        profiles: a list of dicts of word counters to compute IDF for.
        Each counter represents a document:
        [{d1: {w1: count of w1 in d1, w2: count of w2 in d1}, d2: ...}]

    Returns:
        IDFs of the words in documents
    """

    num_docs = sum([len(x) for x in profiles])
    idf = defaultdict(lambda: 0.0)

    for profile in profiles:
        for loc_key in profile:
            for word in profile[loc_key]:
                idf[word] += 1.0

    for word in idf:
        idf[word] = math.log(num_docs / idf[word])

    return idf


def compute_similarities(revs, paps):
    """Compute TPMS similarities

    Args:
        revs: profiles of reviewers
        paps: profiles of papers

    Returns:
        Dict of similarities between each reviewer and paper
    """

    similarities = {}
    idf_dict = compute_idf([revs, paps])

    for r in revs:

        similarities[r] = dict()

        for p in paps:

            r_profile, p_profile = revs[r], paps[p]

            if len(r_profile) == 0 or len(p_profile) == 0:
                raise ValueError(f"Empty profiles in pair {r, p}")

            r_normalizer, p_normalizer = max(r_profile.values()), max(p_profile.values())
            similarity = 0.0

            r_norm, p_norm = 0.0, 0.0

            for word in r_profile:
                r_tf = 0.5 + 0.5 * r_profile[word] / r_normalizer
                w_idf = idf_dict[word]
                r_norm += (r_tf * w_idf) ** 2

                if word in p_profile:
                    p_tf = 0.5 + 0.5 * p_profile[word] / p_normalizer
                    similarity += r_tf * p_tf * (w_idf ** 2)

            for word in p_profile:
                p_tf = 0.5 + 0.5 * p_profile[word] / p_normalizer
                w_idf = idf_dict[word]
                p_norm += (p_tf * w_idf) ** 2

            r_norm, p_norm = math.sqrt(r_norm), math.sqrt(p_norm)

            similarities[r][p] = similarity / r_norm / p_norm

    return similarities

