# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

"""Legal Classification Problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry

import tensorflow as tf

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

_TRAIN_DATASETS = {
    "facts-courts": [
        [
            "https://transfer.sh/iEDNA/gcd.facts-courts.tar.gz",
            ("gcd.facts", "gcd.courts")
        ],
    ],
    "reasons-types": [
        [
            "https://transfer.sh/iEDNA/gcd.facts-courts.tar.gz",
            ("gcd.reasons", "gcd.types")
        ],
    ]
}

_TEST_DATASETS = {
    "facts-courts": [
        [
            "https://transfer.sh/iEDNA/gcd.facts-courts.tar.gz",
            ("gcd.facts-test", "gcd.courts-test")
        ],
    ],
    "reasons-types": [
        [
            "https://transfer.sh/iEDNA/gcd.facts-courts.tar.gz",
            ("gcd.reasons-test", "gcd.types-test")
        ],
    ]
}

_TYPE_CLASSES = ["verdict", "resolution"]
_COURT_CLASSES = ["bag", "bfh", "bgh", "bpatg", "bsg", "bverfg", "bverwg"]

@registry.register_problem
class LegalClassification(problem.Problem):
  """Legal Classification Problem."""

  @property
  def num_shards(self):
    return 10

  @property
  def vocab_file(self):
    return "vocab.class"

  @property
  def targeted_vocab_size(self):
    return 32000

  def doc_generator(self, imdb_dir, dataset, classes, include_label=False):
    for source in sources:
      url = source[0]
      filename = os.path.basename(url)
      compressed_file = maybe_download(tmp_dir, filename, url)

      for file in source[1]:
        tf.logging.info("Reading file: %s" % file)
        filepath = os.path.join(tmp_dir, file)

        # Extract from tar if needed.
        if not tf.gfile.Exists(filepath):
          read_type = "r:gz" if filename.endswith("tgz") else "r"
          with tarfile.open(compressed_file, read_type) as corpus_tar:
            corpus_tar.extractall(tmp_dir)

      with open(source[1][0]) as input_file, open(source[1][1]) as target_file:
          for input, target in zip(input_file, target_file):
              input = input.strip()
              if include_label:
                label = classes.index(target.strip())
                yield doc, label
              else:
                yield doc

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    train_paths = self.training_filepaths(
        data_dir, self.num_shards, shuffled=False)
    dev_paths = self.dev_filepaths(data_dir, 1, shuffled=False)
    generator_utils.generate_dataset_and_shuffle(
        self.generator(data_dir, tmp_dir, True), train_paths,
        self.generator(data_dir, tmp_dir, False), dev_paths)

  @property
  def input_space_id(self):
      return problem.SpaceID.DE_TOK

  @property
  def target_space_id(self):
      return problem.SpaceID.GENERIC

  def example_reading_spec(self):
    data_fields = {
        "inputs": tf.VarLenFeature(tf.int64),
        "targets": tf.FixedLenFeature([1], tf.int64),
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)


@registry.register_problem
class CourtClassification(problem.Problem):
  """Court Classification Problem."""

  @property
  def vocab_file(self):
    return "vocab.multi.class"

  def generator(self, data_dir, tmp_dir, train):
    """Generate examples."""
    # Download and extract
    compressed_filename = os.path.basename(self.URL)
    download_path = generator_utils.maybe_download(tmp_dir, compressed_filename, self.URL)
    gcd_court_dir = os.path.join(tmp_dir, "gcd", "court")
    if not tf.gfile.Exists(imdb_dir):
      with tarfile.open(download_path, "r:gz") as tar:
        tar.extractall(tmp_dir)

    # Generate vocab
    encoder = generator_utils.get_or_generate_vocab_inner(
        data_dir, self.vocab_file, self.targeted_vocab_size,
        self.doc_generator(tmp_dir, _TRAIN_DATASETS["facts-courts"], _COURT_CLASSES))

    # Generate examples
    datasets = _TRAIN_DATASETS["facts-courts"] if train else _TEST_DATASETS["facts-courts"]
    for doc, label in self.doc_generator(tmp_dir, datasets, _COURT_CLASSES, include_label=True):
      yield {
          "inputs": encoder.encode(doc) + [EOS],
          "targets": [label],
      }


  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    source_vocab_size = self._encoders["inputs"].vocab_size
    p.input_modality = {
        "inputs": (registry.Modalities.SYMBOL, source_vocab_size)
    }
    p.target_modality = (registry.Modalities.CLASS_LABEL, len(_COURT_CLASSES))


  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_file)
    encoder = text_encoder.SubwordTextEncoder(vocab_filename)
    return {
        "inputs": encoder,
        "targets": text_encoder.ClassLabelEncoder(_COURT_CLASSES),
    }


@registry.register_problem
class DocumentClassification(problem.Problem):
  """Binary Document Classification Problem."""

  @property
  def vocab_file(self):
    return "vocab.binary.class"

  def generator(self, data_dir, tmp_dir, train):
    """Generate examples."""
    # Download and extract
    compressed_filename = os.path.basename(self.URL)
    download_path = generator_utils.maybe_download(tmp_dir, compressed_filename,
                                                   self.URL)
    gcd_court_dir = os.path.join(tmp_dir, "gcd", "doctype")
    if not tf.gfile.Exists(imdb_dir):
      with tarfile.open(download_path, "r:gz") as tar:
        tar.extractall(tmp_dir)

    # Generate vocab
    encoder = generator_utils.get_or_generate_vocab_inner(
        data_dir, self.vocab_file, self.targeted_vocab_size,
        self.doc_generator(imdb_dir, "train", _TYPE_CLASSES))

    # Generate examples
    dataset = "train" if train else "test"
    for doc, label in self.doc_generator(imdb_dir, dataset, _TYPE_CLASSES, include_label=True):
      yield {
          "inputs": encoder.encode(doc) + [EOS],
          "targets": [label],
      }


  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    source_vocab_size = self._encoders["inputs"].vocab_size
    p.input_modality = {
        "inputs": (registry.Modalities.SYMBOL, source_vocab_size)
    }
    p.target_modality = (registry.Modalities.CLASS_LABEL, len(_TYPE_CLASSES))


  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_file)
    encoder = text_encoder.SubwordTextEncoder(vocab_filename)
    return {
        "inputs": encoder,
        "targets": text_encoder.ClassLabelEncoder(_TYPE_CLASSES),
    }
