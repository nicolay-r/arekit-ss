import argparse
from os.path import join

from arekit.common.data import const
from arekit.common.pipeline.base import BasePipeline
from arekit.common.pipeline.context import PipelineContext
from arekit.contrib.utils.data.writers.csv_native import NativeCsvWriter
from arekit.contrib.utils.data.writers.json_opennre import OpenNREJsonWriter
from arekit.contrib.utils.data.writers.sqlite_native import SQliteWriter
from arekit.contrib.utils.io_utils.samples import SamplesIO

from arekit_ss.entity.masking import StringEntitiesDisplayValueFormatter, MaskedEntitiesFormatter
from arekit_ss.filters.label_type import LabelTextOpinionFilter
from arekit_ss.filters.object_type import EntityBasedTextOpinionFilter
from arekit_ss.framework.samplers_list import create_sampler_pipeline_item
from arekit_ss.sources import src_list
from arekit_ss.sources.config import SourcesConfig
from arekit_ss.text_parser.text_lm import create_lm
from arekit_ss.text_parser.text_nn_ru_frames import create_nn_ru_frames
from arekit_ss.utils import auto_import, setup_custom_logger

text_parsing_pipelines = {
   "nn": create_nn_ru_frames,
   "lm": create_lm
}


if __name__ == '__main__':

    logger = setup_custom_logger("arekit_ss")

    parser = argparse.ArgumentParser(description="Datasource Sampler.")

    parser.add_argument("--writer", type=str, default="csv")
    parser.add_argument("--source", type=str, default="ruattitudes")
    parser.add_argument("--sampler", type=str, default="nn")
    parser.add_argument("--splits", type=str, default=None,
                        help="Manual selection of the data-types related splits that "
                             "should be chosen for the sampling process; types should be "
                             "separated by ':' sign; for example: 'train:test'")
    parser.add_argument("--mask_entities", type=str, default=None, required=False)
    parser.add_argument("--src_lang", type=str, default=None, required=False)
    parser.add_argument("--dest_lang", type=str, default=None, required=False)
    parser.add_argument("--output_dir", type=str, default="_out")
    parser.add_argument("--object-source-types", type=str, default=None,
                        help="Filter specific source object types")
    parser.add_argument("--object-target-types", type=str, default=None,
                        help="Filter specific target object types")
    parser.add_argument("--prompt", type=str, default="{text},`{s_val}`,`{t_val}`, `{label_val}`")
    parser.add_argument("--doc_ids", type=str, default=None)
    parser.add_argument("--relation_types", type=str, default=None,
                        help="list of types, in which items separated with `|` char.")
    parser.add_argument("--docs_limit", type=int, default=None)
    parser.add_argument("--terms_per_context", type=int, default=50)
    parser.add_argument('--no-vectorize', dest='vectorize', action='store_false',
                        help="This flag is applicable only for NN, and denotes "
                             "no need to generate embeddings for features")
    parser.set_defaults(vectorize=True)

    args = parser.parse_args()

    # Setup writer.
    writer = None
    if args.writer == "csv":
        writer = NativeCsvWriter()
    elif args.writer in ['jsonl', 'json']:
        writer = OpenNREJsonWriter(text_columns=["text_a", "text_b"])
    elif args.writer == "sqlite":
        writer = SQliteWriter(skip_existed=False, clear_table=True, index_column_names=[const.ID])
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
    cfg.dest_lang = source["src_lang"] if args.dest_lang is None else args.dest_lang
    cfg.docs_limit = args.docs_limit
    cfg.entities_parser = auto_import(source["entity_parser"], is_class=True)
    cfg.text_parser = text_parsing_pipelines["nn" if args.sampler == "nn" else "lm"](cfg)
    cfg.splits = args.splits
    cfg.do_mask_entities = args.mask_entities is not None

    # Setup filters for text opinions extraction.
    if args.relation_types is not None:
        cfg.optional_filters.append(LabelTextOpinionFilter(
            relation_types=args.relation_types.split("|")))
    if args.object_source_types is not None:
        cfg.optional_filters.append(EntityBasedTextOpinionFilter(
            supported_types=args.object_source_types.split("|"), is_src=True))
    if args.object_target_types is not None:
        cfg.optional_filters.append(EntityBasedTextOpinionFilter(
            supported_types=args.object_target_types.split("|"), is_src=False))

    # Extract data to be serialized in a form of the pipeline.
    dpp = auto_import(name=source["pipeline"])
    data_folding, data_type_pipelines = dpp(cfg)

    # Filter only those data_types that were chosen.
    data_type_pipelines = {k: data_type_pipelines[k] for k in cfg.get_supported_datatypes()
                           if k in data_type_pipelines}

    if len(data_type_pipelines) == 0:
        logger.info(f"DataType Pipelines are empty for the given splits [`{args.splits}`]. No output results.")

    # Forming the name of the result samples by relying on the source name.
    collection_name = "-".join(list(filter(lambda item: item is not None, [
        args.source,
        args.sampler,
        "tpc" + str(args.terms_per_context),
        cfg.dest_lang,
        "l" + str(args.docs_limit) if args.docs_limit is not None else None,
        "fixed" if args.doc_ids is not None else None
    ])))

    pipeline_item = create_sampler_pipeline_item(
        args=args,
        samples_io=SamplesIO(target_dir=args.output_dir, writer=writer, prefix=collection_name),
        label_scaler=auto_import(source["label_scaler"], is_class=True),
        label_fmt=auto_import(source["label_formatter"], is_class=True),
        entity_fmt=StringEntitiesDisplayValueFormatter() if not cfg.do_mask_entities else
            MaskedEntitiesFormatter(subj_mask=args.mask_entities.split(":")[0],
                                    obj_mask=args.mask_entities.split(":")[1],
                                    other_mask=args.mask_entities.split(":")[2]))

    # Launch pipeline.
    pipeline = BasePipeline([pipeline_item])
    pipeline.run(input_data=PipelineContext({
         "doc_ids": args.doc_ids.split(',') if args.doc_ids is not None else None,
         "data_folding": data_folding,
         "data_type_pipelines": data_type_pipelines
     }))

    logger.info(f"Done: {join(args.output_dir, collection_name)} [{args.writer}]")
