from arekit_ss.core.source.brat.entities.parser import BratTextEntitiesParser


class RuAttitudesTextEntitiesParser(BratTextEntitiesParser):

    def __init__(self, text_fmt="list", **kwargs):
        super(RuAttitudesTextEntitiesParser, self).__init__(text_fmt=text_fmt, **kwargs)
