import argparse
from os.path import join

from arekit.contrib.utils.data.writers.csv_pd import PandasCsvWriter
from arekit.contrib.utils.data.writers.json_opennre import OpenNREJsonWriter

import sources.s_ruattitudes as s_ra
import sources.s_rusentrel as s_rsr
import sources.s_sentinerel as s_snrL


sources = {
    "ruattitudes": {
        "nn": s_ra.do_serialize_nn,
        "bert": s_ra.do_serialize_bert
    },
    "rusentrel": {
        "nn": s_rsr.do_serialize_nn,
        "bert": s_rsr.do_serialize_bert
    },
    "sentinerel": {
        "nn": s_snrL.do_serialize_nn,
        "bert": s_snrL.do_serialize_bert
    }
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Source Translator.")

    parser.add_argument("--writer", type=str, default="csv")
    parser.add_argument("--source", type=str, default="ruattitudes")
    parser.add_argument("--sampler", type=str, default="nn")
    parser.add_argument("--dest_lang", type=str, default="en")
    parser.add_argument("--output_dir", type=str, default="_out")
    parser.add_argument("--docs_limit", type=int, default=1)

    args = parser.parse_args()

    # Completing the output template.
    output_dir = join(args.output_dir, '-'.join([args.source, args.dest_lang,
                                                 str(args.docs_limit) if args.docs_limit is not None else "all"]))

    # Setup writer.
    writer = None
    if args.writer == "csv":
        writer = PandasCsvWriter(write_header=True)
    elif args.writer == "jsonl":
        writer = OpenNREJsonWriter(text_columns=["text_a", "text_b"])

    # Running handler
    handler = sources[args.source][args.sampler]
    handler(writer=writer, dest_lang=args.dest_lang, docs_limit=args.docs_limit, output_dir=output_dir)
