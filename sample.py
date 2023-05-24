import argparse
from os.path import join

from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.utils.data.writers.csv_native import NativeCsvWriter
from arekit.contrib.utils.data.writers.json_opennre import OpenNREJsonWriter

import sources.s_ruattitudes as s_ra
import sources.s_rusentrel as s_rsr
import sources.s_sentinerel as s_snr
from framework.arekit.rows_bert import create_bert_rows_provider
from framework.arekit.rows_nn import create_nn_rows_provider
from framework.arekit.serialize_bert import serialize_bert_pipeline
from framework.arekit.serialize_nn import serialize_nn_pipeline
from sources.config import SourcesConfig
from sources.scaler import PosNegNeuRelationsLabelScaler

data_provider_pipelines = {
    "ruattitudes": {
        "nn": s_ra.build_datapipeline_nn,
        "bert": s_ra.build_datapipeline_bert
    },
    "rusentrel": {
        "nn": s_rsr.build_datapipeline_nn,
        "bert": s_rsr.build_datapipeline_bert
    },
    "sentinerel": {
        "nn": s_snr.build_datapipeline_nn,
        "bert": s_snr.build_datapipeline_bert
    }
}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Source Translator.")

    parser.add_argument("--writer", type=str, default="csv")
    parser.add_argument("--source", type=str, default="ruattitudes")
    parser.add_argument("--sampler", type=str, default="nn")
    parser.add_argument("--dest_lang", type=str, default="en")
    parser.add_argument("--output_dir", type=str, default="_out")
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

    # Extract data to be serialized in a form of the pipeline.
    dpp = data_provider_pipelines[args.source][args.sampler]
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
            sample_row_provider=create_bert_rows_provider(
                terms_per_context=args.terms_per_context,
                labels_scaler=labels_scaler))

    # Launch pipeline.
    pipeline = BasePipeline([pipeline_item])
    pipeline.run(input_data=None, params_dict={
                     "data_folding": data_folding,
                     "data_type_pipelines": data_type_pipelines
                 })
