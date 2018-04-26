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
import tarfile
import tensorflow as tf

FLAGS = tf.flags.FLAGS

EOS = text_encoder.EOS_ID

_TRAIN_DATASETS = {
    "cs":
        [
            "https://transfer.sh/clezv/jrc_acquis_summarize.tar.gz",
            ("jrc_acquis.cs.fulltexts", "jrc_acquis.cs.summaries")
        ],
    "de":
        [
            "https://transfer.sh/clezv/jrc_acquis_summarize.tar.gz",
            ("jrc_acquis.de.fulltexts", "jrc_acquis.de.summaries")
        ],

    "en":
        [
            "https://transfer.sh/clezv/jrc_acquis_summarize.tar.gz",
            ("jrc_acquis.en.fulltexts", "jrc_acquis.en.summaries")
        ],

    "es":
        [
            "https://transfer.sh/clezv/jrc_acquis_summarize.tar.gz",
            ("jrc_acquis.es.fulltexts", "jrc_acquis.es.summaries")
        ],

    "fr":
        [
            "https://transfer.sh/clezv/jrc_acquis_summarize.tar.gz",
            ("jrc_acquis.fr.fulltexts", "jrc_acquis.fr.summaries")
        ],

    "it":
        [
            "https://transfer.sh/clezv/jrc_acquis_summarize.tar.gz",
            ("jrc_acquis.it.fulltexts", "jrc_acquis.it.summaries")
        ],

    "sv":
        [
            "https://transfer.sh/clezv/jrc_acquis_summarize.tar.gz",
            ("jrc_acquis.sv.fulltexts", "jrc_acquis.sv.summaries")
        ]
}

_TEST_DATASETS = {
    "cs":
        [
            "https://transfer.sh/pqvnr/jrc_acquis.summarize-test.tar.gz",
            ("jrc_acquis.cs-test.fulltexts", "jrc_acquis.cs-test.summaries")
        ],

    "de":
        [
            "https://transfer.sh/pqvnr/jrc_acquis.summarize-test.tar.gz",
            ("jrc_acquis.de-test.fulltexts", "jrc_acquis.de-test.summaries")
        ],

    "en":
        [
            "https://transfer.sh/pqvnr/jrc_acquis.summarize-test.tar.gz",
            ("jrc_acquis.en-test.fulltexts", "jrc_acquis.en-test.summaries")
        ],
    "es":
    [
            "https://transfer.sh/pqvnr/jrc_acquis.summarize-test.tar.gz",
            ("jrc_acquis.es-test.fulltexts",
             "jrc_acquis.es-test.summaries")
        ],

    "fr":
        [
            "https://transfer.sh/pqvnr/jrc_acquis.summarize-test.tar.gz",
            ("jrc_acquis.fr-test.fulltexts", "jrc_acquis.fr-test.summaries")
        ],

    "it":
        [
            "https://transfer.sh/pqvnr/jrc_acquis.summarize-test.tar.gz",
            ("jrc_acquis.it-test.fulltexts", "jrc_acquis.it-test.summaries")
        ],

    "sv":
        [
            "https://transfer.sh/pqvnr/jrc_acquis.summarize-test.tar.gz",
            ("jrc_acquis.sv-test.fulltexts", "jrc_acquis.sv-test.summaries")
        ]

}


def download_and_extract_data(tmp_dir, dataset):
    """Download and Extract files."""
    url = dataset[0]
    print(dataset[0])
    print(dataset[0])
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

    fulltexts_filename, summaries_filename = dataset[1]
    fulltexts_filepath = os.path.join(tmp_dir, fulltexts_filename)
    summaries_filepath = os.path.join(tmp_dir, summaries_filename)
    return fulltexts_filepath, summaries_filepath


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
class SummarizeLegal32k(problem.Text2TextProblem):
    """Summarize jrc aquis docs according to their head section"""

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
            metrics.Metrics.ACC_PER_SEQ, metrics.Metrics.NEG_LOG_PERPLEXITY,
            metrics.Metrics.ROUGE_1_F, metrics.Metrics.ROUGE_2_F, metrics.Metrics.ROUGE_L_F
        ]


@registry.register_problem
class SummarizeCsLegal32k(SummarizeLegal32k):
    """Summarize cs documents"""

    @property
    def input_space_id(self):
        return problem.SpaceID.CS_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.CS_TOK

    @property
    def vocab_name(self):
        return "vocab.sum.cs"

    def generator(self, data_dir, tmp_dir, train):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, [_TRAIN_DATASETS["cs"]])
        datasets = _TRAIN_DATASETS["cs"] if train else _TEST_DATASETS["cs"]
        fulltext_file, summaries_file = download_and_extract_data(
            tmp_dir, datasets)
        return token_generator(fulltext_file, summaries_file, vocab, EOS)


@registry.register_problem
class SummarizeDeLegal32k(SummarizeLegal32k):
    """Summarize de documents"""

    @property
    def input_space_id(self):
        return problem.SpaceID.DE_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.DE_TOK

    @property
    def vocab_name(self):
        return "vocab.sum.de"

    def generator(self, data_dir, tmp_dir, train):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, [_TRAIN_DATASETS["de"]])
        datasets = _TRAIN_DATASETS["de"] if train else _TEST_DATASETS["de"]
        fulltext_file, summaries_file = download_and_extract_data(
            tmp_dir, datasets)
        return token_generator(fulltext_file, summaries_file, vocab, EOS)


@registry.register_problem
class SummarizeEnLegal32k(SummarizeLegal32k):
    """Summarize en documents"""

    @property
    def input_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def vocab_name(self):
        return "vocab.sum.en"

    def generator(self, data_dir, tmp_dir, train):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, [_TRAIN_DATASETS["en"]])
        datasets = _TRAIN_DATASETS["en"] if train else _TEST_DATASETS["en"]
        fulltext_file, summaries_file = download_and_extract_data(
            tmp_dir, datasets)
        return token_generator(fulltext_file, summaries_file, vocab, EOS)


@registry.register_problem
class SummarizeEsLegal32k(SummarizeLegal32k):
    """Summarize es documents"""

    @property
    def input_space_id(self):
        return problem.SpaceID.ES_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.ES_TOK

    @property
    def vocab_name(self):
        return "vocab.sum.es"

    def generator(self, data_dir, tmp_dir, train):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, [_TRAIN_DATASETS["es"]])
        datasets = _TRAIN_DATASETS["es"] if train else _TEST_DATASETS["es"]
        fulltext_file, summaries_file = download_and_extract_data(
            tmp_dir, datasets)
        return token_generator(fulltext_file, summaries_file, vocab, EOS)


@registry.register_problem
class SummarizeFrLegal32k(SummarizeLegal32k):
    """Summarize fr documents"""

    @property
    def input_space_id(self):
        return problem.SpaceID.FR_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.FR_TOK

    @property
    def vocab_name(self):
        return "vocab.sum.fr"

    def generator(self, data_dir, tmp_dir, train):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, [_TRAIN_DATASETS["fr"]])
        datasets = _TRAIN_DATASETS["fr"] if train else _TEST_DATASETS["fr"]
        fulltext_file, summaries_file = download_and_extract_data(
            tmp_dir, datasets)
        return token_generator(fulltext_file, summaries_file, vocab, EOS)


@registry.register_problem
class SummarizeItLegal32k(SummarizeLegal32k):
    """Summarize it documents"""

    @property
    def input_space_id(self):
        return problem.SpaceID.IT_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.IT_TOK

    @property
    def vocab_name(self):
        return "vocab.sum.it"

    def generator(self, data_dir, tmp_dir, train):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, [_TRAIN_DATASETS["it"]])
        datasets = _TRAIN_DATASETS["it"] if train else _TEST_DATASETS["it"]
        fulltext_file, summaries_file = download_and_extract_data(
            tmp_dir, datasets)
        return token_generator(fulltext_file, summaries_file, vocab, EOS)


@registry.register_problem
class SummarizeSvLegal32k(SummarizeLegal32k):
    """Summarize sv documents"""

    @property
    def input_space_id(self):
        return problem.SpaceID.SV_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.SV_TOK

    @property
    def vocab_name(self):
        return "vocab.sum.sv"

    def generator(self, data_dir, tmp_dir, train):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, [_TRAIN_DATASETS["sv"]])
        datasets = _TRAIN_DATASETS["sv"] if train else _TEST_DATASETS["sv"]
        fulltext_file, summaries_file = download_and_extract_data(
            tmp_dir, datasets)
        return token_generator(fulltext_file, summaries_file, vocab, EOS)
