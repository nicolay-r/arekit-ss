from arekit.common.experiment.data_type import DataType


class SourcesConfig:

    def __init__(self):
        self.terms_per_context = 50
        self.src_lang = "ru"
        self.dest_lang = "en"
        self.docs_limit = None
        self.entities_parser = None
        self.text_parser = None
        self.splits = None
        self.optional_filters = []

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
    

