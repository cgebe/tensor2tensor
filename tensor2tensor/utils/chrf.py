# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ChrF metric util used during eval for MT.
Reference:
Maja Popović (2015). chrF: character n-gram F-score for automatic MT evaluation. In Proceedings of the Tenth Workshop on Statistical Machine Translationn, pages 392–395, Lisbon, Portugal.
http://www.statmt.org/wmt15/pdf/WMT49.pdf
originally by: https://github.com/rsennrich/subword-nmt/blob/master/chrF.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
# pylint: disable=redefined-builtin
from six.moves import zip
# pylint: enable=redefined-builtin

import tensorflow as tf

def character_ngram_f3score(predictions, labels, **unused_kwargs):
    """ChrF3 score computation between labels and predictions.

    Args:
      predictions: tensor, model predictions
      labels: tensor, gold output.
    Returns:
      chrF3: int, approx chrF3 score
    """

    outputs = tf.to_int32(tf.argmax(predictions, axis=-1))
    # Convert the outputs and labels to a [batch_size, input_length] tensor.
    outputs = tf.squeeze(outputs, axis=[-1, -2])
    labels = tf.squeeze(labels, axis=[-1, -2])

    chrF = tf.py_func(compute_chrF, (labels, outputs), tf.float32)

    return chrF, tf.constant(1.0)


def compute_chrF(reference_corpus,
                 translation_corpus,
                 recall_importance=3,
                 max_order=6):
    """Computes ChrF score of translated segments against one or more references.

    Args:
      reference_corpus: list of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing ChrF score. 6 is default.

    Returns:
      ChrF score.
    """

    total_ref = [0] * max_order
    total_hyp = [0] * max_order
    correct_total_hyp = [0] * max_order
    for (references, translations) in zip(reference_corpus, translation_corpus):
        ngrams_ref = extract_ngrams(references, max_order)
        ngrams_hyp = extract_ngrams(translations, max_order)

        add_ngrams_ref(ngrams_ref, total_ref)
        add_ngrams_hyp(ngrams_ref, ngrams_hyp, total_hyp, correct_total_hyp)

    return fscore(total_ref, total_hyp, correct_total_hyp, max_order, recall_importance)


def add_ngrams_ref(ngrams_ref, total_ref):
    """Extracts correct ngrams count (count of an ngram which occurs in the reference
    and in the hypothesis) and the total count of ngrams (ngrams which occur in the hypothesis).
    Args:
      ngrams_ref: ngrams dict for the reference
      ngrams_hyp: ngrams dict for the hypothesis
      max_order: maximum length of ngrams.
    Returns:
      correct_total_hyp, total_hyp: respective arrays stating the counts for each ngram rank
    """
    for rank in ngrams_ref:
        for chain in ngrams_ref[rank]:
            total_ref[rank] += ngrams_ref[rank][chain]

    return total_ref


def add_ngrams_hyp(ngrams_ref, ngrams_hyp, total_hyp, correct_total_hyp):
    """Extracts correct ngrams count (count of an ngram which occurs in the reference
    and in the hypothesis) and the total count of ngrams (ngrams which occur in the hypothesis).
    Args:
      ngrams_ref: ngrams dict for the reference
      ngrams_hyp: ngrams dict for the hypothesis
      max_order: maximum length of ngrams.
    Returns:
      correct_total_hyp, total_hyp: respective arrays stating the counts for each ngram rank
    """
    for rank in ngrams_hyp:
        for chain in ngrams_hyp[rank]:
            total_hyp[rank] += ngrams_hyp[rank][chain]
            if chain in ngrams_ref[rank]:
                correct_total_hyp[rank] += min(ngrams_hyp[rank][chain],
                                     ngrams_ref[rank][chain])

    return total_hyp, correct_total_hyp


def fscore(total_ref, total_hyp, correct_total_hyp, max_order, recall_importance, smooth=0):
    """Computes the final fscore according to ngram precision and recall.
    Args:
      correct_total_hyp: count for each rank how many ngrams matched
      total_hyp: count for each rank how many ngrams in hypthesis
      total_ref: count for each rank how many ngrams in reference
      max_order: maximum length of ngrams.
    Returns:
      correct_total_hyp, total_hyp: respective arrays stating the counts for each ngram rank
    """

    precision = 0
    recall = 0

    for i in range(max_order):
        if total_hyp[i] + smooth and total_ref[i] + smooth:
            precision += (correct_total_hyp[i] + smooth) / (total_hyp[i] + smooth)
            recall += (correct_total_hyp[i] + smooth) / (total_ref[i] + smooth)

    precision /= max_order
    recall /= max_order

    return (1 + recall_importance**2) * (precision * recall) / ((recall_importance**2 * precision) + recall)


def extract_ngrams(segment, max_order, spaces=False):
    """Extracts all n-grams up to a given maximum order from an input segment.
    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length of ngrams.
      spaces: whether spaces should be included or squashed
    Returns:
      results: results according to rank, how often a specific ngram occured in the segment
    """
    if not spaces:
        segment = ''.join(segment.split())
    else:
        segment = segment.strip()

    results = defaultdict(lambda: defaultdict(int))
    for length in range(max_order):
        for start_pos in range(len(segment)):
            end_pos = start_pos + length + 1
            if end_pos <= len(segment):
                results[length][tuple(segment[start_pos: end_pos])] += 1

    return results
