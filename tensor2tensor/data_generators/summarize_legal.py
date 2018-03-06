

@registry.register_problem
class SummarizeJrcAquis32k(problem.Text2TextProblem):
    """Summarize german court decision according to their tenor"""

    @property
    def is_character_level(self):
        return False

    @property
    def has_inputs(self):
        return True

    @property
    def input_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def num_shards(self):
        return 100

    @property
    def vocab_name(self):
        return "vocab.cnndailymail"

    @property
    def use_subword_tokenizer(self):
        return True

    @property
    def targeted_vocab_size(self):
        return 2**15  # 32768

    @property
    def use_train_shards_for_dev(self):
        return False

    def generator(self, data_dir, tmp_dir, is_training):
        all_files, urls_path = _maybe_download_corpora(tmp_dir, is_training)
        encoder = generator_utils.get_or_generate_vocab_inner(
            data_dir, self.vocab_file, self.targeted_vocab_size,
            example_generator(all_files, urls_path, sum_token=False))
        write_raw_text_to_files(all_files, urls_path, data_dir, tmp_dir,
                                is_training)
        for example in example_generator(all_files, urls_path, sum_token=True):
            story, summary = _story_summary_split(example)
            encoded_summary = encoder.encode(summary) + [EOS]
            encoded_story = encoder.encode(story) + [EOS]
            yield {"inputs": encoded_story, "targets": encoded_summary}


@registry.register_problem
class SummarizeJrcAquis32k(problem.Text2TextProblem):
    """Summarize Jrc documents according to their head sentences"""

    @property
    def is_character_level(self):
        return False

    @property
    def has_inputs(self):
        return True

    @property
    def input_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def num_shards(self):
        return 100

    @property
    def vocab_name(self):
        return "vocab.cnndailymail"

    @property
    def use_subword_tokenizer(self):
        return True

    @property
    def targeted_vocab_size(self):
        return 2**15  # 32768

    @property
    def use_train_shards_for_dev(self):
        return False

    def generator(self, data_dir, tmp_dir, is_training):
        all_files, urls_path = _maybe_download_corpora(tmp_dir, is_training)
        encoder = generator_utils.get_or_generate_vocab_inner(
            data_dir, self.vocab_file, self.targeted_vocab_size,
            example_generator(all_files, urls_path, sum_token=False))
        write_raw_text_to_files(all_files, urls_path, data_dir, tmp_dir,
                                is_training)
        for example in example_generator(all_files, urls_path, sum_token=True):
            story, summary = _story_summary_split(example)
            encoded_summary = encoder.encode(summary) + [EOS]
            encoded_story = encoder.encode(story) + [EOS]
            yield {"inputs": encoded_story, "targets": encoded_summary}
