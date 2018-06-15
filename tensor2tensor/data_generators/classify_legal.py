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
import numpy

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow as tf

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

_TRAIN_DATASETS = {
    "facts-courts": [
        [
            "https://transfer.sh/VMXIr/gcd.facts-courts.tar.gz",
            ("gcd.facts", "gcd.courts")
        ],
    ],
    "facts-result": [
        [
            "https://transfer.sh/4LGW1/facts-result.tar.gz",
            ("gcd.facts", "gcd.results")
        ],
    ]
}

_TEST_DATASETS = {
    "facts-courts": [
        [
            "https://transfer.sh/VMXIr/gcd.facts-courts.tar.gz",
            ("gcd.facts-test", "gcd.courts-test")
        ],
    ],
    "facts-result": [
        [
            "https://transfer.sh/4LGW1/facts-result.tar.gz",
            ("gcd.facts-test", "gcd.results-test")
        ],
    ]
}

_RESULT_CLASSES = ["positive", "negative"]
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

    def doc_generator(self, tmp_dir, datasets, classes, include_label=False):
        label_encoder = text_encoder.ClassLabelEncoder(class_labels=classes)
        for source in datasets:
            url = source[0]
            filename = os.path.basename(url)
            compressed_file = generator_utils.maybe_download(
                tmp_dir, filename, url)

            for file in source[1]:
                tf.logging.info("Reading file: %s" % file)
                filepath = os.path.join(tmp_dir, file)

                # Extract from tar if needed.
                if not tf.gfile.Exists(filepath):
                    with tarfile.open(compressed_file, "r:gz") as corpus_tar:
                        corpus_tar.extractall(tmp_dir)

            with open(os.path.join(tmp_dir, source[1][0])) as input_file, open(os.path.join(tmp_dir, source[1][1])) as target_file:
                for input, target in zip(input_file, target_file):
                    input = input.strip()
                    if include_label:
                        yield input, label_encoder.encode(target.strip())
                    else:
                        yield input

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
            "targets": tf.VarLenFeature(tf.int64),
        }
        data_items_to_decoders = None
        return (data_fields, data_items_to_decoders)

    def eval_metrics(self):
        return [
            metrics.Metrics.SIGMOID_ACCURACY_ONE_HOT, metrics.Metrics.SIGMOID_PRECISION_ONE_HOT,
            metrics.Metrics.SIGMOID_RECALL_ONE_HOT,
            metrics.Metrics.NEG_LOG_PERPLEXITY
        ]


@registry.register_problem
class CourtClassification(LegalClassification):
    """Court Classification Problem."""

    @property
    def vocab_file(self):
        return "vocab.multi.class"

    def generator(self, data_dir, tmp_dir, train):
        """Generate examples."""
        # Generate vocab
        encoder = generator_utils.get_or_generate_vocab_inner(
            data_dir, self.vocab_file, self.targeted_vocab_size,
            self.doc_generator(tmp_dir, _TRAIN_DATASETS["facts-courts"], _COURT_CLASSES))

        # Generate examples
        datasets = _TRAIN_DATASETS["facts-courts"] if train else _TEST_DATASETS["facts-courts"]
        for doc, label in self.doc_generator(tmp_dir, datasets, _COURT_CLASSES, include_label=True):
            print(encoder.encode(doc))
            print([label])
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
        p.input_space_id = self.input_space_id
        p.target_space_id = self.target_space_id

    def feature_encoders(self, data_dir):
        vocab_filename = os.path.join(data_dir, self.vocab_file)
        encoder = text_encoder.SubwordTextEncoder(vocab_filename)
        return {
            "inputs": encoder,
            "targets": text_encoder.ClassLabelEncoder(class_labels=_COURT_CLASSES),
        }

@registry.register_problem
class VerdictClassification(LegalClassification):
    """Binary Verdict Classification Problem."""

    @property
    def vocab_file(self):
        return "vocab.verdict.class"

    def generator(self, data_dir, tmp_dir, train):
        """Generate examples."""
        # Generate vocab
        encoder = generator_utils.get_or_generate_vocab_inner(
            data_dir, self.vocab_file, self.targeted_vocab_size,
            self.doc_generator(tmp_dir, _TRAIN_DATASETS["facts-result"], _RESULT_CLASSES))

        # Generate examples
        datasets = _TRAIN_DATASETS["facts-result"] if train else _TEST_DATASETS["facts-result"]
        for doc, label in self.doc_generator(tmp_dir, datasets, _RESULT_CLASSES, include_label=True):
            print(label, end=" ")
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
        p.target_modality = (registry.Modalities.CLASS_LABEL, len(_RESULT_CLASSES))

    def feature_encoders(self, data_dir):
        vocab_filename = os.path.join(data_dir, self.vocab_file)
        encoder = text_encoder.SubwordTextEncoder(vocab_filename)
        return {
            "inputs": encoder,
            "targets": text_encoder.ClassLabelEncoder(_RESULT_CLASSES),
        }
