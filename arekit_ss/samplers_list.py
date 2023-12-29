from arekit.contrib.utils.io_utils.embedding import NpEmbeddingIO

from arekit_ss.core.rows_bert import create_bert_rows_provider
from arekit_ss.core.rows_prompt import create_prompt_rows_provider
from arekit_ss.core.rows_ru_sentiment_nn import create_ru_sentiment_nn_rows_provider
from arekit_ss.core.serialize_bert import serialize_bert_pipeline
from arekit_ss.core.serialize_nn import serialize_nn_pipeline
from arekit_ss.sources.labels.scaler_frames import ThreeLabelScaler


def create_sampler_pipeline_item(args, samples_io, label_scaler, label_fmt, entity_fmt):
    """ This function represent a factory of all the potential samplers,
        oriented for sentiment analysis task (labels_scaler)
    """

    if "nn" == args.sampler:
        return serialize_nn_pipeline(
            samples_io=samples_io,
            emb_io=NpEmbeddingIO(target_dir=args.output_dir, prefix_name=samples_io.Prefix),
            rows_provider=create_ru_sentiment_nn_rows_provider(
                relation_labels_scaler=label_scaler,
                frame_roles_label_scaler=ThreeLabelScaler(),
                vectorizers="default" if args.vectorize else None,
                entity_fmt=entity_fmt))

    elif "bert" == args.sampler:
        return serialize_bert_pipeline(
            samples_io=samples_io,
            rows_provider=create_bert_rows_provider(
                terms_per_context=args.terms_per_context,
                labels_scaler=label_scaler,
                entity_fmt=entity_fmt))

    elif "prompt" == args.sampler:
        # same as for BERT.
        return serialize_bert_pipeline(
            samples_io=samples_io,
            rows_provider=create_prompt_rows_provider(
                prompt=args.prompt,
                labels_scaler=label_scaler,
                # We consider a default labels formatter.
                labels_formatter=label_fmt,
                entity_fmt=entity_fmt))
