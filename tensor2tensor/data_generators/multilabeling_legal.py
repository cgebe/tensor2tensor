MultiLabeling# coding=utf-8
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

import tensorflow as tf

FLAGS = tf.flags.FLAGS

EOS = text_encoder.EOS_ID

_TRAIN_DATASETS = {
    "cs": [
        [
            "https://transfer.sh/J95xx/jrc_acquis_multi_labeling.tar.gz",
            ("jrc_acquis.cs.fulltexts", "jrc_acquis.cs.eurovoc")
        ],
    ],
    "de": [
        [
            "https://transfer.sh/J95xx/jrc_acquis_multi_labeling.tar.gz",
            ("jrc_acquis.de.fulltexts", "jrc_acquis.de.eurovoc")
        ],
    ],
    "en": [
        [
            "https://transfer.sh/J95xx/jrc_acquis_multi_labeling.tar.gz",
            ("jrc_acquis.en.fulltexts", "jrc_acquis.en.eurovoc")
        ],
    ],
    "es": [
        [
            "https://transfer.sh/J95xx/jrc_acquis_multi_labeling.tar.gz",
            ("jrc_acquis.es.fulltexts", "jrc_acquis.es.eurovoc")
        ],
    ],
    "fr": [
        [
            "https://transfer.sh/J95xx/jrc_acquis_multi_labeling.tar.gz",
            ("jrc_acquis.fr.fulltexts", "jrc_acquis.fr.eurovoc")
        ],
    ],
    "it": [
        [
            "https://transfer.sh/J95xx/jrc_acquis_multi_labeling.tar.gz",
            ("jrc_acquis.it.fulltexts", "jrc_acquis.it.eurovoc")
        ],
    ],
    "sv": [
        [
            "https://transfer.sh/J95xx/jrc_acquis_multi_labeling.tar.gz",
            ("jrc_acquis.sv.fulltexts", "jrc_acquis.sv.eurovoc")
        ],
    ],
}

_TEST_DATASETS = {
    "cs": [
        [
            "https://transfer.sh/11fBci/jrc_acquis_multi_labeling-test.tar.gz",
            ("jrc_acquis.cs-test.fulltexts", "jrc_acquis.cs-test.eurovoc")
        ],
    ],
    "de": [
        [
            "https://transfer.sh/11fBci/jrc_acquis_multi_labeling-test.tar.gz",
            ("jrc_acquis.de-test.fulltexts", "jrc_acquis.de-test.eurovoc")
        ],
    ],
    "en": [
        [
            "https://transfer.sh/11fBci/jrc_acquis_multi_labeling-test.tar.gz",
            ("jrc_acquis.en-test.fulltexts", "jrc_acquis.en-test.eurovoc")
        ],
    ],
    "fr": [
        [
            "https://transfer.sh/11fBci/jrc_acquis_multi_labeling-test.tar.gz",
            ("jrc_acquis.fr-test.fulltexts", "jrc_acquis.fr-test.eurovoc")
        ],
    ],
    "it": [
        [
            "https://transfer.sh/11fBci/jrc_acquis_multi_labeling-test.tar.gz",
            ("jrc_acquis.it-test.fulltexts", "jrc_acquis.it-test.eurovoc")
        ],
    ],
    "sv": [
        [
            "https://transfer.sh/11fBci/jrc_acquis_multi_labeling-test.tar.gz",
            ("jrc_acquis.sv-test.fulltexts", "jrc_acquis.sv-test.eurovoc")
        ],
    ]
}


def compile_data(tmp_dir, datasets, filename):
    """Concatenate all `datasets` and save to `filename`."""
    filename = os.path.join(tmp_dir, filename)
    with tf.gfile.GFile(filename + ".fulltexts", mode="w") as fulltexts_resfile:
        with tf.gfile.GFile(filename + ".eurovoc", mode="w") as eurovoc_resfile:
            for dataset in datasets:
                url = dataset[0]
                compressed_filename = os.path.basename(url)
                compressed_filepath = os.path.join(
                    tmp_dir, compressed_filename)

                generator_utils.maybe_download(
                    tmp_dir, compressed_filename, url)

                fulltexts_filename, eurovoc_filename = dataset[1]
                fulltexts_filepath = os.path.join(tmp_dir, fulltexts_filename)
                eurovoc_filepath = os.path.join(tmp_dir, eurovoc_filename)

                if not (os.path.exists(fulltexts_filepath) and
                        os.path.exists(eurovoc_filepath)):
                    # For .tar.gz and .tgz files, we read compressed.
                    mode = "r:gz" if compressed_filepath.endswith(
                        "gz") else "r"
                    with tarfile.open(compressed_filepath, mode) as corpus_tar:
                        corpus_tar.extractall(tmp_dir)
                if fulltexts_filepath.endswith(".gz"):
                    new_filepath = fulltexts_filepath.strip(".gz")
                    generator_utils.gunzip_file(
                        fulltexts_filepath, new_filepath)
                    fulltexts_filepath = new_filepath
                if eurovoc_filepath.endswith(".gz"):
                    new_filepath = eurovoc_filepath.strip(".gz")
                    generator_utils.gunzip_file(
                        eurovoc_filepath, new_filepath)
                    eurovoc_filepath = new_filepath

                with tf.gfile.GFile(fulltexts_filepath, mode="r") as fulltexts_file:
                    with tf.gfile.GFile(eurovoc_filepath, mode="r") as eurovoc_file:
                        line1, line2 = fulltexts_file.readline(), eurovoc_file.readline()
                        while line1 or line2:
                            line1res = _preprocess_sgm(line1, False)
                            line2res = _preprocess_sgm(line2, False)
                            if line1res or line2res:
                                fulltexts_resfile.write(
                                    line1res.strip() + "\n")
                                eurovoc_resfile.write(
                                    line2res.strip() + "\n")
                            line1, line2 = fulltexts_file.readline(), eurovoc_file.readline()

    return filename


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
                target_ints = target.strip().split(" ") + eos_list
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
        return 100

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

    def generator(self, data_dir, tmp_dir, is_training):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["cs"])
        datasets = _TRAIN_DATASETS["cs"] if train else _TEST_DATASETS["cs"]
        tag = "train" if train else "dev"
        # compile to save the texts onto disc
        data_path = compile_data(
            tmp_dir, datasets, "multi_labeling_cs_tok_%s" % tag)
        return token_generator(data_path + ".fulltexts", data_path + ".eurovoc", vocab, EOS)


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

    def generator(self, data_dir, tmp_dir, is_training):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["de"])
        datasets = _TRAIN_DATASETS["de"] if train else _TEST_DATASETS["de"]
        tag = "train" if train else "dev"
        # compile to save the texts onto disc
        data_path = compile_data(
            tmp_dir, datasets, "multi_labeling_de_tok_%s" % tag)
        return token_generator(data_path + ".fulltexts", data_path + ".eurovoc", vocab, EOS)


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

    def generator(self, data_dir, tmp_dir, is_training):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["en"])
        datasets = _TRAIN_DATASETS["en"] if train else _TEST_DATASETS["en"]
        tag = "train" if train else "dev"
        # compile to save the texts onto disc
        data_path = compile_data(
            tmp_dir, datasets, "multi_labeling_en_tok_%s" % tag)
        return token_generator(data_path + ".fulltexts", data_path + ".eurovoc", vocab, EOS)


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

    def generator(self, data_dir, tmp_dir, is_training):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["es"])
        datasets = _TRAIN_DATASETS["es"] if train else _TEST_DATASETS["es"]
        tag = "train" if train else "dev"
        # compile to save the texts onto disc
        data_path = compile_data(
            tmp_dir, datasets, "multi_labeling_es_tok_%s" % tag)
        return token_generator(data_path + ".fulltexts", data_path + ".eurovoc", vocab, EOS)


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

    def generator(self, data_dir, tmp_dir, is_training):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["fr"])
        datasets = _TRAIN_DATASETS["fr"] if train else _TEST_DATASETS["fr"]
        tag = "train" if train else "dev"
        # compile to save the texts onto disc
        data_path = compile_data(
            tmp_dir, datasets, "multi_labeling_fr_tok_%s" % tag)
        return token_generator(data_path + ".fulltexts", data_path + ".eurovoc", vocab, EOS)


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

    def generator(self, data_dir, tmp_dir, is_training):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["it"])
        datasets = _TRAIN_DATASETS["it"] if train else _TEST_DATASETS["it"]
        tag = "train" if train else "dev"
        # compile to save the texts onto disc
        data_path = compile_data(
            tmp_dir, datasets, "multi_labeling_it_tok_%s" % tag)
        return token_generator(data_path + ".fulltexts", data_path + ".eurovoc", vocab, EOS)


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

    def generator(self, data_dir, tmp_dir, is_training):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["sv"])
        datasets = _TRAIN_DATASETS["sv"] if train else _TEST_DATASETS["sv"]
        tag = "train" if train else "dev"
        # compile to save the texts onto disc
        data_path = compile_data(
            tmp_dir, datasets, "multi_labeling_sv_tok_%s" % tag)
        return token_generator(data_path + ".fulltexts", data_path + ".eurovoc", vocab, EOS)
