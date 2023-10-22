from arekit.common.experiment.data_type import DataType
from arekit.contrib.utils.pipelines.items.text.translator import MLTextTranslatorPipelineItem

from arekit_ss.third_party.utils import create_translate_model


class SourcesConfig:

    def __init__(self):
        self.terms_per_context = 50
        self.src_lang = None
        self.dest_lang = None
        self.translator_backend = "googletrans"
        self.docs_limit = None
        self.entities_parser = None
        self.text_parser = None
        self.splits = None
        self.do_mask_entites = False
        self.optional_filters = []

    def get_translator_pipeline_item(self, do_translation):

        translate_model = {
            None: lambda: None,
            "googletrans": lambda: create_translate_model(backend="googletrans")
        }

        # Setup translator.
        translator = translate_model["googletrans" if do_translation else None]()

        text_translator_setup = {
            None: lambda: None,
            "ml-based": lambda: MLTextTranslatorPipelineItem(
                batch_translate_model=lambda content: translator(
                    str_list=content, src=self.src_lang, dest=self.dest_lang),
                do_translate_entity=not self.do_mask_entites)
        }

        return text_translator_setup["ml-based" if do_translation else None]()

    def get_supported_datatypes(self):
        """ String split name to data-types converter.
        """

        # AREkit 0.24.0 has the predefined type DataType which describes the
        # splits in a form of Enum.
        data_type_to_split = {
            "train": DataType.Train,
            "test": DataType.Test,
            "dev": DataType.Dev,
            "etalon": DataType.Etalon
        }

        if self.splits is None:
            return set(data_type_to_split.values())

        chosen_splits = set(self.splits.split(":"))
        return set([data_type_to_split[split_name] for split_name in chosen_splits])
    

