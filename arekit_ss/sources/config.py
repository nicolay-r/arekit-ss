class SourcesConfig:

    def __init__(self):
        self.terms_per_context = 50
        self.src_lang = "ru"
        self.dest_lang = "en"
        self.docs_limit = None
        self.entities_parser = None
        self.text_parser = None
