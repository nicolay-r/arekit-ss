import argparse
from os.path import join

from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.utils.data.writers.csv_native import NativeCsvWriter
from arekit.contrib.utils.data.writers.json_opennre import OpenNREJsonWriter

from arekit_ss.framework.samplers_list import create_sampler_pipeline_item
from arekit_ss.sources import src_list
from arekit_ss.sources.config import SourcesConfig
from arekit_ss.text_parser.text_lm import create_lm
from arekit_ss.text_parser.text_nn_ru_frames import create_nn_ru_frames

text_parsing_pipelines = {
   "nn": create_nn_ru_frames,
   "lm": create_lm
}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Datasource Sampler.")

    parser.add_argument("--writer", type=str, default="csv")
    parser.add_argument("--source", type=str, default="ruattitudes")
    parser.add_argument("--sampler", type=str, default="nn")
    parser.add_argument("--src_lang", type=str, default=None, required=False)
    parser.add_argument("--dest_lang", type=str, default="en")
    parser.add_argument("--output_dir", type=str, default="_out")
    parser.add_argument("--prompt", type=str, default="{text},`{s_val}`,`{t_val}`, `{label_val}`")
    parser.add_argument("--text_parser", type=str, default="nn")
    parser.add_argument("--docs_limit", type=int, default=None)
    parser.add_argument("--terms_per_context", type=int, default=50)
    parser.add_argument('--no-vectorize', dest='vectorize', action='store_false',
                        help="This flag is applicable only for NN, and denotes "
                             "no need to generate embeddings for features")
    parser.set_defaults(vectorize=True)

    args = parser.parse_args()

    # Completing the output template.
    output_dir = join(args.output_dir, '-'.join([args.source, args.dest_lang,
                                                 str(args.docs_limit) if args.docs_limit is not None else "all"]))

    # Setup writer.
    writer = None
    if args.writer == "csv":
        writer = NativeCsvWriter()
    elif args.writer in ['jsonl', 'json']:
        writer = OpenNREJsonWriter(text_columns=["text_a", "text_b"])
    else:
        raise Exception("writer `{}` is not supported!".format(args.writer))

    source = src_list.DATA_PROVIDER_PIPELINES[args.source]

    # Provide the check for the source language.
    if args.src_lang is not None and "src_lang" in source:
        assert(source["src_lang"] == args.src_lang)

    # Initialize config.
    cfg = SourcesConfig()
    cfg.terms_per_context = args.terms_per_context
    cfg.src_lang = source["src_lang"] if args.src_lang is None else args.src_lang
    cfg.dest_lang = args.dest_lang
    cfg.docs_limit = args.docs_limit
    cfg.entities_parser = source["entity_parser"]
    cfg.text_parser = text_parsing_pipelines[args.text_parser](cfg)

    # Extract data to be serialized in a form of the pipeline.
    dpp = source["pipeline"]
    data_folding, data_type_pipelines = dpp(cfg)

    # Prepare serializer and pass data_type_pipelines.
    pipeline_item = create_sampler_pipeline_item(
        args=args, writer=writer,
        label_scaler=source["label_scaler"],
        label_fmt=source["label_formatter"])

    # Launch pipeline.
    pipeline = BasePipeline([pipeline_item])
    pipeline.run(input_data=None, params_dict={
                     "data_folding": data_folding,
                     "data_type_pipelines": data_type_pipelines
                 })
