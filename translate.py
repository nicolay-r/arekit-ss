import argparse

from arekit.contrib.utils.data.writers.csv_pd import PandasCsvWriter
from arekit.contrib.utils.data.writers.json_opennre import OpenNREJsonWriter

from sources.s_ruattitudes import do_serialize_bert, do_serialize_nn

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Source Translator.")
    parser.add_argument("--writer", type=str)
    parser.add_argument("--sampler", type=str)
    parser.add_argument("--dest_lang", type=str, default="en")
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    # Setup writer.
    writer = None
    if args.writer == "csv":
        writer = PandasCsvWriter(write_header=True)
    elif args.writer == "jsonl":
        writer = OpenNREJsonWriter(text_columns=["text_a", "text_b"])

    # Choose Serializer.
    if args.sampler == "bert":
        do_serialize_bert(writer=writer, dest_lang=args.dest_lang, limit=args.limit)
    elif args.sampler == "nn":
        do_serialize_nn(writer=writer, dest_lang=args.dest_lang, limit=args.limit)
