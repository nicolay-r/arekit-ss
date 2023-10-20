from arekit.common.entities.base import Entity
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.entities.types import OpinionEntityType


class MaskedEntitiesFormatter(StringEntitiesFormatter):

    def __init__(self, obj_mask="#0", subj_mask="#S", other_mask="#E"):
        self.__obj_mask = obj_mask
        self.__subj_mask = subj_mask
        self.__other_mask = other_mask

    def to_string(self, original_value, entity_type):
        assert(isinstance(entity_type, OpinionEntityType))

        if (entity_type == OpinionEntityType.Object) or (entity_type == OpinionEntityType.SynonymObject):
            return self.__obj_mask
        elif (entity_type == OpinionEntityType.Subject) or (entity_type == OpinionEntityType.SynonymSubject):
            return self.__subj_mask
        elif entity_type == OpinionEntityType.Other:
            return self.__other_mask


class StringEntitiesDisplayValueFormatter(StringEntitiesFormatter):
    """ Provides the contents of the DisplayValue property.
    """

    def to_string(self, original_value, entity_type):
        assert(isinstance(original_value, Entity))
        return original_value.DisplayValue
