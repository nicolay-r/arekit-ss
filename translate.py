import argparse

from arekit.contrib.utils.data.writers.csv_pd import PandasCsvWriter
from arekit.contrib.utils.data.writers.json_opennre import OpenNREJsonWriter

import sources.s_ruattitudes as s_ra
import sources.s_rusentrel as s_rsr


sources = {
    "ruattitudes": {
        "nn": s_ra.do_serialize_nn,
        "bert": s_ra.do_serialize_bert
    },
    "rusentrel": {
        "nn": s_rsr.do_serialize_nn,
        "bert": s_rsr.do_serialize_bert
    }
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Source Translator.")

    parser.add_argument("--writer", type=str)
    parser.add_argument("--source", type=str)
    parser.add_argument("--sampler", type=str)
    parser.add_argument("--dest_lang", type=str, default="en")
    parser.add_argument("--output_dir_template", type=str, default="_out/serialize")
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    # Completing the output template.
    output_dir = '-'.join([args.output_dir_template, args.sampler, args.source])

    # Setup writer.
    writer = None
    if args.writer == "csv":
        writer = PandasCsvWriter(write_header=True)
    elif args.writer == "jsonl":
        writer = OpenNREJsonWriter(text_columns=["text_a", "text_b"])

    # Running handler
    handler = sources[args.source][args.sampler]
    handler(writer=writer, dest_lang=args.dest_lang, limit=args.limit, output_dir=output_dir)
