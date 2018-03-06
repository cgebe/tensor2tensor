"""Data generators for summarization of jrc_acquis"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

import tensorflow as tf

FLAGS = tf.flags.FLAGS

EOS = text_encoder.EOS_ID

_TRAIN_DATASETS = {
    "cs": [
        [
            "https://transfer.sh/RPGgI/jrc_acquis_summarization.tar.gz",
            ("jrc_acquis.cs.fulltexts", "jrc_acquis.cs.summaries")
        ],
    ],
    "de": [
        [
            "https://transfer.sh/RPGgI/jrc_acquis_summarization.tar.gz",
            ("jrc_acquis.de.fulltexts", "jrc_acquis.de.summaries")
        ],
    ],
    "en": [
        [
            "https://transfer.sh/RPGgI/jrc_acquis_summarization.tar.gz",
            ("jrc_acquis.en.fulltexts", "jrc_acquis.en.summaries")
        ],
    ],
    "es": [
        [
            "https://transfer.sh/RPGgI/jrc_acquis_summarization.tar.gz",
            ("jrc_acquis.es.fulltexts", "jrc_acquis.es.summaries")
        ],
    ],
    "fr": [
        [
            "https://transfer.sh/RPGgI/jrc_acquis_summarization.tar.gz",
            ("jrc_acquis.fr.fulltexts", "jrc_acquis.fr.summaries")
        ],
    ],
    "it": [
        [
            "https://transfer.sh/RPGgI/jrc_acquis_summarization.tar.gz",
            ("jrc_acquis.it.fulltexts", "jrc_acquis.it.summaries")
        ],
    ],
    "sv": [
        [
            "https://transfer.sh/RPGgI/jrc_acquis_summarization.tar.gz",
            ("jrc_acquis.sv.fulltexts", "jrc_acquis.sv.summaries")
        ],
    ],
}

_TEST_DATASETS = {
    "cs": [
        [
            "https://transfer.sh/rfFbv/jrc_acquis_summarization-test.tar.gz",
            ("jrc_acquis.cs-test.fulltexts", "jrc_acquis.cs-test.summaries")
        ],
    ],
    "de": [
        [
            "https://transfer.sh/rfFbv/jrc_acquis_summarization-test.tar.gz",
            ("jrc_acquis.de-test.fulltexts", "jrc_acquis.de-test.summaries")
        ],
    ],
    "en": [
        [
            "https://transfer.sh/rfFbv/jrc_acquis_summarization-test.tar.gz",
            ("jrc_acquis.en-test.fulltexts", "jrc_acquis.en-test.summaries")
        ],
    ],
    "fr": [
        [
            "https://transfer.sh/rfFbv/jrc_acquis_summarization-test.tar.gz",
            ("jrc_acquis.fr-test.fulltexts", "jrc_acquis.fr-test.summaries")
        ],
    ],
    "it": [
        [
            "https://transfer.sh/rfFbv/jrc_acquis_summarization-test.tar.gz",
            ("jrc_acquis.it-test.fulltexts", "jrc_acquis.it-test.summaries")
        ],
    ],
    "sv": [
        [
            "https://transfer.sh/rfFbv/jrc_acquis_summarization-test.tar.gz",
            ("jrc_acquis.sv-test.fulltexts", "jrc_acquis.sv-test.summaries")
        ],
    ]
}


def compile_data(tmp_dir, datasets, filename):
    """Concatenate all `datasets` and save to `filename`."""
    filename = os.path.join(tmp_dir, filename)
    with tf.gfile.GFile(filename + ".fulltexts", mode="w") as fulltexts_resfile:
        with tf.gfile.GFile(filename + ".summaries", mode="w") as summaries_resfile:
            for dataset in datasets:
                url = dataset[0]
                compressed_filename = os.path.basename(url)
                compressed_filepath = os.path.join(
                    tmp_dir, compressed_filename)

                generator_utils.maybe_download(
                    tmp_dir, compressed_filename, url)

                fulltexts_filename, summaries_filename = dataset[1]
                fulltexts_filepath = os.path.join(tmp_dir, fulltexts_filename)
                summaries_filepath = os.path.join(tmp_dir, summaries_filename)

                if not (os.path.exists(fulltexts_filepath) and
                        os.path.exists(summaries_filepath)):
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
                if summaries_filepath.endswith(".gz"):
                    new_filepath = summaries_filepath.strip(".gz")
                    generator_utils.gunzip_file(
                        summaries_filepath, new_filepath)
                    summaries_filepath = new_filepath

                with tf.gfile.GFile(fulltexts_filepath, mode="r") as fulltexts_file:
                    with tf.gfile.GFile(summaries_filepath, mode="r") as summaries_file:
                        line1, line2 = fulltexts_file.readline(), summaries_file.readline()
                        while line1 or line2:
                            line1res = _preprocess_sgm(line1, False)
                            line2res = _preprocess_sgm(line2, False)
                            if line1res or line2res:
                                fulltexts_resfile.write(
                                    line1res.strip() + "\n")
                                summaries_resfile.write(
                                    line2res.strip() + "\n")
                            line1, line2 = fulltexts_file.readline(), summaries_file.readline()

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
        return 100

    @property
    def use_subword_tokenizer(self):
        return True

    @property
    def targeted_vocab_size(self):
        return 2**15  # 32768

    @property
    def use_train_shards_for_dev(self):
        return False


@registry.register_problem
class SummarizeCsLegal32k(summarize_legal.SummarizeLegal32kProblem):
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

    def generator(self, data_dir, tmp_dir, is_training):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["cs"])
        datasets = _TRAIN_DATASETS["cs"] if train else _TEST_DATASETS["cs"]
        tag = "train" if train else "dev"
        # compile to save the texts onto disc
        data_path = summarize_legal.compile_data(
            tmp_dir, datasets, "summarize_cs_tok_%s" % tag)
        return summarize_legal.token_generator(data_path + ".fulltexts", data_path + ".summaries", vocab, EOS)


@registry.register_problem
class SummarizeDeLegal32k(summarize_legal.SummarizeLegal32kProblem):
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

    def generator(self, data_dir, tmp_dir, is_training):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["de"])
        datasets = _TRAIN_DATASETS["de"] if train else _TEST_DATASETS["de"]
        tag = "train" if train else "dev"
        # compile to save the texts onto disc
        data_path = summarize_legal.compile_data(
            tmp_dir, datasets, "summarize_de_tok_%s" % tag)
        return summarize_legal.token_generator(data_path + ".fulltexts", data_path + ".summaries", vocab, EOS)


@registry.register_problem
class SummarizeEnLegal32k(summarize_legal.SummarizeLegal32kProblem):
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

    def generator(self, data_dir, tmp_dir, is_training):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["en"])
        datasets = _TRAIN_DATASETS["en"] if train else _TEST_DATASETS["en"]
        tag = "train" if train else "dev"
        # compile to save the texts onto disc
        data_path = summarize_legal.compile_data(
            tmp_dir, datasets, "summarize_en_tok_%s" % tag)
        return summarize_legal.token_generator(data_path + ".fulltexts", data_path + ".summaries", vocab, EOS)


@registry.register_problem
class SummarizeEsLegal32k(summarize_legal.SummarizeLegal32kProblem):
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

    def generator(self, data_dir, tmp_dir, is_training):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["es"])
        datasets = _TRAIN_DATASETS["es"] if train else _TEST_DATASETS["es"]
        tag = "train" if train else "dev"
        # compile to save the texts onto disc
        data_path = summarize_legal.compile_data(
            tmp_dir, datasets, "summarize_es_tok_%s" % tag)
        return summarize_legal.token_generator(data_path + ".fulltexts", data_path + ".summaries", vocab, EOS)


@registry.register_problem
class SummarizeFrLegal32k(summarize_legal.SummarizeLegal32kProblem):
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

    def generator(self, data_dir, tmp_dir, is_training):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["fr"])
        datasets = _TRAIN_DATASETS["fr"] if train else _TEST_DATASETS["fr"]
        tag = "train" if train else "dev"
        # compile to save the texts onto disc
        data_path = summarize_legal.compile_data(
            tmp_dir, datasets, "summarize_fr_tok_%s" % tag)
        return summarize_legal.token_generator(data_path + ".fulltexts", data_path + ".summaries", vocab, EOS)


@registry.register_problem
class SummarizeItLegal32k(summarize_legal.SummarizeLegal32kProblem):
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

    def generator(self, data_dir, tmp_dir, is_training):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["it"])
        datasets = _TRAIN_DATASETS["it"] if train else _TEST_DATASETS["it"]
        tag = "train" if train else "dev"
        # compile to save the texts onto disc
        data_path = summarize_legal.compile_data(
            tmp_dir, datasets, "summarize_it_tok_%s" % tag)
        return summarize_legal.token_generator(data_path + ".fulltexts", data_path + ".summaries", vocab, EOS)


@registry.register_problem
class SummarizeSvLegal32k(summarize_legal.SummarizeLegal32kProblem):
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

    def generator(self, data_dir, tmp_dir, is_training):
        vocab = generator_utils.get_or_generate_vocab(
            data_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["sv"])
        datasets = _TRAIN_DATASETS["sv"] if train else _TEST_DATASETS["sv"]
        tag = "train" if train else "dev"
        # compile to save the texts onto disc
        data_path = summarize_legal.compile_data(
            tmp_dir, datasets, "summarize_sv_tok_%s" % tag)
        return summarize_legal.token_generator(data_path + ".fulltexts", data_path + ".summaries", vocab, EOS)
