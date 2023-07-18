from arekit_ss.framework.arekit.rows_bert import create_bert_rows_provider
from arekit_ss.framework.arekit.rows_prompt import create_prompt_rows_provider
from arekit_ss.framework.arekit.rows_ru_sentiment_nn import create_ru_sentiment_nn_rows_provider
from arekit_ss.framework.arekit.serialize_bert import serialize_bert_pipeline
from arekit_ss.framework.arekit.serialize_nn import serialize_nn_pipeline

from arekit_ss.sources.labels.formatter import PosNegNeuLabelsFormatter
from arekit_ss.sources.labels.scaler import PosNegNeuRelationsLabelScaler
from arekit_ss.sources.labels.scaler_frames import ThreeLabelScaler


def create_sentiment_sampler_pipeline_item(args, writer):
    """ This function represent a factory of all the potential samplers,
        oriented for sentiment analysis task (labels_scaler)
    """

    relation_labels_scaler = PosNegNeuRelationsLabelScaler()

    if "nn" == args.sampler:
        return serialize_nn_pipeline(
            output_dir=args.output_dir, writer=writer,
            rows_provider=create_ru_sentiment_nn_rows_provider(
                relation_labels_scaler=relation_labels_scaler,
                frame_roles_label_scaler=ThreeLabelScaler(),
                vectorizers="default" if args.vectorize else None))

    elif "bert" == args.sampler:
        return serialize_bert_pipeline(
            output_dir=args.output_dir, writer=writer,
            rows_provider=create_bert_rows_provider(
                terms_per_context=args.terms_per_context,
                labels_scaler=relation_labels_scaler))

    elif "prompt" == args.sampler:
        # same as for BERT.
        text_labels_formatter = PosNegNeuLabelsFormatter()
        return serialize_bert_pipeline(
            output_dir=args.output_dir, writer=writer,
            rows_provider=create_prompt_rows_provider(
                prompt=args.prompt,
                labels_scaler=relation_labels_scaler,
                # We consider a default labels formatter.
                labels_formatter=text_labels_formatter))
