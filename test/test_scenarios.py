import os
import unittest


class TestScenarios(unittest.TestCase):

    def test_sentiment_analysis_frames_annotation(self, src="rusentrel"):
        # The case when we do not adopt the translation.
        cmd = "python3 -m arekit_ss.sample --writer csv "\
              "--source {src} --sampler nn --src_lang ru --dest_lang ru "\
              "--docs_limit 1 --text_parser lm --output_dir './_out/{src}_nn'".replace("{src}", src)
        os.system(cmd)

    def test_sentiment_analysis(self, src="rusentrel"):
        cmd = "python -m arekit_ss.sample --writer csv --source {src} " \
              "--sampler bert --src_lang ru --dest_lang ru --docs_limit 1 --text_parser lm "\
              "--output_dir './_out/{src}_bert'".replace("{src}", src)
        os.system(cmd)

    def test_sentiment_analysis_translation_prompt(self, src="rusentrel"):
        cmd = "python -m arekit_ss.sample --writer csv --source {src} --sampler prompt "\
              "--prompt \"For text: '{text}', the attitude between '{s_val}' and '{t_val}' is: '{label_val}'\" "\
              "--src_lang ru --dest_lang ru --docs_limit 1 --text_parser lm"\
              "--output_dir './_out/{src}_prompt'".replace("{src}", src)
        os.system(cmd)
