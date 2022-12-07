from arekit.common.entities.base import Entity


class EntityHelper(object):
    """ Форматирование типов сущностей в тексте.
    """

    # Ключи
    AGE = "AGE"
    AWARD = "AWARD"
    CITY = "CITY"
    COUNTRY = "COUNTRY"
    CRIME = "CRIME"
    DATE = "DATE"
    DISEASE = "DISEASE"
    DISTRICT = "DISTRICT"
    EVENT = "EVENT"
    FACILITY = "FACILITY"
    FAMILY = "FAMILY"
    IDEOLOGY = "IDEOLOGY"
    LANGUAGE = "LANGUAGE"
    LAW = "LAW"
    LOCATION = "LOCATION"
    MONEY = "MONEY"
    NATIONALITY = "NATIONALITY"
    NUMBER = "NUMBER"
    ORDINAL = "ORDINAL"
    ORGANIZATION = "ORGANIZATION"
    PENALTY = "PENALTY"
    PERCENT = "PERCENT"
    PERSON = "PERSON"
    PRODUCT = "PRODUCT"
    PROFESSION = "PROFESSION"
    RELIGION = "RELIGION"
    STATE_OR_PROVINCE = "STATE_OR_PROVINCE"
    TIME = "TIME"
    WORK_OF_ART = "WORK_OF_ART"

    __types_fmt = {
        AGE: "возраст",
        AWARD: "награда",
        CITY: "город",
        COUNTRY: "страна",
        CRIME: "преступление",
        DATE: "дата",
        DISEASE: "болезнь",
        DISTRICT: "район",
        EVENT: "событие",
        FACILITY: "сооружение",
        FAMILY: "семья",
        IDEOLOGY: "идеология",
        LANGUAGE: "язык",
        LAW: "закон",
        LOCATION: "локация",
        MONEY: "средства",
        NATIONALITY: "национальность",
        NUMBER: "количество",
        ORDINAL: "номер",
        ORGANIZATION: "организация",
        PENALTY: "штраф",
        PERCENT: "процент",
        PERSON: "личность",
        PRODUCT: "продукт",
        PROFESSION: "профессия",
        RELIGION: "религия",
        STATE_OR_PROVINCE: "штат",
        TIME: "время",
        WORK_OF_ART: "исскуство"
    }

    # Можно полагаться на BERT-ontonotes, в котором поддерживаются следующие типы:
    # http://docs.deeppavlov.ai/en/master/features/models/ner.html#named-entity-recognition-ner
    __supported_list = [
        # AGE
        # AWARD
        # CRIME
        # DISTRICT
        # FAMILY,
        # IDEOLOGY,
        # PENALTY,
        # RELIGION,
        PROFESSION,
        NATIONALITY,            # NORP
        DATE,
        STATE_OR_PROVINCE,      # GPE-like
        LAW,
        LANGUAGE,
        LOCATION,
        MONEY,
        ORGANIZATION,
        PERCENT,
        PERSON,
        PRODUCT,
        EVENT,
        CITY,                   # LOCATION-like
        COUNTRY,                # LOCATION-like
        ORDINAL,
        NUMBER,
        FACILITY,
        TIME,
        WORK_OF_ART
    ]

    __supported_set = set(__supported_list)

    @staticmethod
    def format(entity):
        assert(isinstance(entity, Entity))
        return EntityHelper.__types_fmt[entity.Type] \
            if entity.Type in EntityHelper.__supported_set \
            else entity.Value
