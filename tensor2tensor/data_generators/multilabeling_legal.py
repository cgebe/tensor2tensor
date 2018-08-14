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

"""Data generators for summarization of jrc_acquis"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import os
import tensorflow as tf

FLAGS = tf.flags.FLAGS

EOS = text_encoder.EOS_ID

_TRAIN_DATASETS = {
    "cs":
        [
            "https://transfer.sh/JIxJs/jrc_acquis.label.tar.gz",
            ("jrc_acquis.cs.documents", "jrc_acquis.cs.labels")
        ],
    "de":
        [
            "https://transfer.sh/JIxJs/jrc_acquis.label.tar.gz",
            ("jrc_acquis.de.documents", "jrc_acquis.de.labels")
        ],
    "en":
        [
            "https://transfer.sh/JIxJs/jrc_acquis.label.tar.gz",
            ("jrc_acquis.en.documents", "jrc_acquis.en.labels")
        ],
    "es":
        [
            "https://transfer.sh/JIxJs/jrc_acquis.label.tar.gz",
            ("jrc_acquis.es.documents", "jrc_acquis.es.labels")
        ],
    "fr":
        [
            "https://transfer.sh/JIxJs/jrc_acquis.label.tar.gz",
            ("jrc_acquis.fr.documents", "jrc_acquis.fr.labels")
        ],
    "it":
        [
            "https://transfer.sh/JIxJs/jrc_acquis.label.tar.gz",
            ("jrc_acquis.it.documents", "jrc_acquis.it.labels")
        ],
    "sv":
        [
            "https://transfer.sh/JIxJs/jrc_acquis.label.tar.gz",
            ("jrc_acquis.sv.documents", "jrc_acquis.sv.labels")
        ],
}

_TEST_DATASETS = {
    "cs":
        [
            "https://transfer.sh/g8Y8s/jrc_acquis.label-test.tar.gz",
            ("jrc_acquis.cs-test.documents", "jrc_acquis.cs-test.labels")
        ],
    "de":
        [
            "https://transfer.sh/g8Y8s/jrc_acquis.label-test.tar.gz",
            ("jrc_acquis.de-test.documents", "jrc_acquis.de-test.labels")
        ],
    "en":
        [
            "https://transfer.sh/g8Y8s/jrc_acquis.label-test.tar.gz",
            ("jrc_acquis.en-test.documents", "jrc_acquis.en-test.labels")
        ],
    "es":
        [
            "https://transfer.sh/g8Y8s/jrc_acquis.label-test.tar.gz",
            ("jrc_acquis.es-test.documents", "jrc_acquis.es-test.labels")
        ],
    "fr":
        [
            "https://transfer.sh/g8Y8s/jrc_acquis.label-test.tar.gz",
            ("jrc_acquis.fr-test.documents", "jrc_acquis.fr-test.labels")
        ],
    "it":
        [
            "https://transfer.sh/g8Y8s/jrc_acquis.label-test.tar.gz",
            ("jrc_acquis.it-test.documents", "jrc_acquis.it-test.labels")
        ],
    "sv":
        [
            "https://transfer.sh/g8Y8s/jrc_acquis.label-test.tar.gz",
            ("jrc_acquis.sv-test.documents", "jrc_acquis.sv-test.labels")
        ],

}


def download_and_extract_data(tmp_dir, dataset):
    """Download and Extract files."""
    url = dataset[0]
    print(dataset)
    compressed_filename = os.path.basename(url)
    compressed_file = generator_utils.maybe_download(
        tmp_dir, compressed_filename, url)

    for file in dataset[1]:
        tf.logging.info("Reading file: %s" % file)
        filepath = os.path.join(tmp_dir, file)

        # Extract from tar if needed.
        if not tf.gfile.Exists(filepath):
            with tarfile.open(compressed_file, "r:gz") as corpus_tar:
                corpus_tar.extractall(tmp_dir)

    documents_filename, labels_filename = dataset[1]
    documents_filepath = os.path.join(tmp_dir, documents_filename)
    labels_filepath = os.path.join(tmp_dir, labels_filename)
    return documents_filepath, labels_filepath


def token_generator(source_path, target_path, token_vocab, eos=None):
    """Generator for sequence-to-sequence tasks that uses tokens.

    This generator assumes the files at source_path and target_path have
    the same number of lines and yields dictionaries of "inputs" and "targets"
    where inputs are token ids from the " "-split source (and target, resp.) lines
    converted to integers using the token_map.

    Args:
      source_path: path to the file with source sentences.
      target_path: path to the file with target sentences.
      token_vocab: text_encoder.TextEncoder object.
      eos: integer to append at the end of each sequence (default: None).
    Yields:
      A dictionary {"inputs": source-line, "targets": target-line} where
      the lines are integer lists converted from tokens in the file lines.
    """
    eos_list = [] if eos is None else [eos]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            while source and target:
                source_ints = token_vocab.encode(source.strip()) + eos_list
                target_ints = token_vocab.encode(target.strip()) + eos_list
                yield {"inputs": source_ints, "targets": target_ints}
                source, target = source_file.readline(), target_file.readline()


@registry.register_problem
class MultiLabelingLegal32k(problem.Text2TextProblem):
    """MultiLabeling jrc aquis docs according to their head section"""

    @property
    def is_character_level(self):
        return False

    @property
    def has_inputs(self):
        return True

    @property
    def num_shards(self):
        return 10

    @property
    def use_subword_tokenizer(self):
        return True

    @property
    def targeted_vocab_size(self):
        return 32000

    @property
    def use_train_shards_for_dev(self):
        return False

    def eval_metrics(self):
        return [
            metrics.Metrics.ACC, metrics.Metrics.ACC_TOP5,
            metrics.Metrics.ACC_PER_SEQ, metrics.Metrics.NEG_LOG_PERPLEXITY
        ]


@registry.register_problem
class MultiLabelingCsLegal32k(MultiLabelingLegal32k):
    """MultiLabeling cs documents"""

    @property
    def input_space_id(self):
        return problem.SpaceID.CS_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.GENERIC

    @property
    def vocab_name(self):
        return "vocab.labeling.cs"

    def generator(self, data_dir, tmp_dir, train):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, [_TRAIN_DATASETS["cs"]])
        datasets = _TRAIN_DATASETS["cs"] if train else _TEST_DATASETS["cs"]
        document_file, labels_file = download_and_extract_data(
            tmp_dir, datasets)
        return token_generator(document_file, labels_file, vocab, EOS)


@registry.register_problem
class MultiLabelingDeLegal32k(MultiLabelingLegal32k):
    """MultiLabeling de documents"""

    @property
    def input_space_id(self):
        return problem.SpaceID.DE_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.GENERIC

    @property
    def vocab_name(self):
        return "vocab.labeling.de"

    def generator(self, data_dir, tmp_dir, train):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, [_TRAIN_DATASETS["de"]])
        datasets = _TRAIN_DATASETS["de"] if train else _TEST_DATASETS["de"]
        document_file, labels_file = download_and_extract_data(
            tmp_dir, datasets)
        return token_generator(document_file, labels_file, vocab, EOS)


@registry.register_problem
class MultiLabelingEnLegal32k(MultiLabelingLegal32k):
    """MultiLabeling en documents"""

    @property
    def input_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.GENERIC

    @property
    def vocab_name(self):
        return "vocab.labeling.en"

    def generator(self, data_dir, tmp_dir, train):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, [_TRAIN_DATASETS["en"]])
        datasets = _TRAIN_DATASETS["en"] if train else _TEST_DATASETS["en"]
        document_file, labels_file = download_and_extract_data(
            tmp_dir, datasets)
        return token_generator(document_file, labels_file, vocab, EOS)


@registry.register_problem
class MultiLabelingEsLegal32k(MultiLabelingLegal32k):
    """MultiLabeling es documents"""

    @property
    def input_space_id(self):
        return problem.SpaceID.ES_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.GENERIC

    @property
    def vocab_name(self):
        return "vocab.labeling.es"

    def generator(self, data_dir, tmp_dir, train):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, [_TRAIN_DATASETS["es"]])
        datasets = _TRAIN_DATASETS["es"] if train else _TEST_DATASETS["es"]
        document_file, labels_file = download_and_extract_data(
            tmp_dir, datasets)
        return token_generator(document_file, labels_file, vocab, EOS)


@registry.register_problem
class MultiLabelingFrLegal32k(MultiLabelingLegal32k):
    """MultiLabeling fr documents"""

    @property
    def input_space_id(self):
        return problem.SpaceID.FR_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.GENERIC

    @property
    def vocab_name(self):
        return "vocab.labeling.fr"

    def generator(self, data_dir, tmp_dir, train):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, [_TRAIN_DATASETS["fr"]])
        datasets = _TRAIN_DATASETS["fr"] if train else _TEST_DATASETS["fr"]
        document_file, labels_file = download_and_extract_data(
            tmp_dir, datasets)
        return token_generator(document_file, labels_file, vocab, EOS)


@registry.register_problem
class MultiLabelingItLegal32k(MultiLabelingLegal32k):
    """MultiLabeling it documents"""

    @property
    def input_space_id(self):
        return problem.SpaceID.IT_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.GENERIC

    @property
    def vocab_name(self):
        return "vocab.labeling.it"

    def generator(self, data_dir, tmp_dir, train):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, [_TRAIN_DATASETS["it"]])
        datasets = _TRAIN_DATASETS["it"] if train else _TEST_DATASETS["it"]
        document_file, labels_file = download_and_extract_data(
            tmp_dir, datasets)
        return token_generator(document_file, labels_file, vocab, EOS)


@registry.register_problem
class MultiLabelingSvLegal32k(MultiLabelingLegal32k):
    """MultiLabeling sv documents"""

    @property
    def input_space_id(self):
        return problem.SpaceID.SV_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.GENERIC

    @property
    def vocab_name(self):
        return "vocab.labeling.sv"

    def generator(self, data_dir, tmp_dir, train):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, [_TRAIN_DATASETS["sv"]])
        datasets = _TRAIN_DATASETS["sv"] if train else _TEST_DATASETS["sv"]
        document_file, labels_file = download_and_extract_data(
            tmp_dir, datasets)
        return token_generator(document_file, labels_file, vocab, EOS)
