import argparse
from os.path import join

from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.utils.data.writers.csv_native import NativeCsvWriter
from arekit.contrib.utils.data.writers.json_opennre import OpenNREJsonWriter

from arekit_ss.framework.arekit.rows_bert import create_bert_rows_provider
from arekit_ss.framework.arekit.rows_nn import create_nn_rows_provider
from arekit_ss.framework.arekit.rows_prompt import create_prompt_rows_provider
from arekit_ss.framework.arekit.serialize_bert import serialize_bert_pipeline
from arekit_ss.framework.arekit.serialize_nn import serialize_nn_pipeline
from arekit_ss.sources import src_list
from arekit_ss.sources.config import SourcesConfig
from arekit_ss.sources.labels.scaler import PosNegNeuRelationsLabelScaler
from arekit_ss.text_parser.text_lm import create_lm
from arekit_ss.text_parser.text_nn_frames import create_nn_frames


text_parsing_pipelines = {
   "nn-frames": create_nn_frames,
   "lm": create_lm
}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Datasource Sampler.")

    parser.add_argument("--writer", type=str, default="csv")
    parser.add_argument("--source", type=str, default="ruattitudes")
    parser.add_argument("--sampler", type=str, default="nn")
    parser.add_argument("--dest_lang", type=str, default="en")
    parser.add_argument("--output_dir", type=str, default="_out")
    parser.add_argument("--prompt", type=str, default="{text},`{s_ind}`,`{t_ind}`, `{label}`")
    parser.add_argument("--text_parser", type=str, default="nn")
    parser.add_argument("--docs_limit", type=int, default=None)
    parser.add_argument("--terms_per_context", type=int, default=50)

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

    # Initialize config.
    cfg = SourcesConfig()
    cfg.terms_per_context = args.terms_per_context
    cfg.dest_lang = args.dest_lang
    cfg.docs_limit = args.docs_limit
    cfg.entities_parser = src_list.ENTITY_PARSERS[args.source]
    cfg.text_parser = text_parsing_pipelines[args.text_parser](cfg)

    # Extract data to be serialized in a form of the pipeline.
    dpp = src_list.DATA_PROVIDER_PIPELINES[args.source]
    data_folding, data_type_pipelines = dpp(cfg)

    labels_scaler = PosNegNeuRelationsLabelScaler()

    # Prepare serializer and pass data_type_pipelines.
    pipeline_item = None
    if "nn" == args.sampler:
        pipeline_item = serialize_nn_pipeline(
            output_dir=args.output_dir, writer=writer,
            rows_provider=create_nn_rows_provider(labels_scaler))
    elif "bert" == args.sampler:
        pipeline_item = serialize_bert_pipeline(
            output_dir=args.output_dir, writer=writer,
            rows_provider=create_bert_rows_provider(
                terms_per_context=args.terms_per_context,
                labels_scaler=labels_scaler))
    elif "prompt" == args.sampler:
        # same as for BERT.
        pipeline_item = serialize_bert_pipeline(
            output_dir=args.output_dir, writer=writer,
            rows_provider=create_prompt_rows_provider(
                prompt=args.prompt, labels_scaler=labels_scaler))

    # Launch pipeline.
    pipeline = BasePipeline([pipeline_item])
    pipeline.run(input_data=None, params_dict={
                     "data_folding": data_folding,
                     "data_type_pipelines": data_type_pipelines
                 })
