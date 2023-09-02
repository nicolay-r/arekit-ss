import logging
import time
from googletrans import Translator

from arekit.common.data.input.providers.const import IDLE_MODE
from arekit.common.pipeline.conts import PARENT_CTX

from arekit.common.entities.base import Entity
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)


class TextAndEntitiesGoogleTranslator(BasePipelineItem):
    """ Text translator, based on GoogleTranslate service
        NOTE: Considered to be adopted right-after entities parsed
        but before the input tokenization into list of terms.
        For entities we update and assign its DisplayValue.
        NOTE#2: Move this pipeline item as a separated github project.
    """

    def __init__(self, src, dest, attempts=10, timeout_for_connection_lost_sec=1.0):
        assert(isinstance(src, str))
        assert(isinstance(src, str))
        self.translator = Translator()
        self.__src = src
        self.__dest = dest
        self.__attempts = attempts
        self.__timeout_for_connection_lost = timeout_for_connection_lost_sec

    def fast_most_accurate_approach(self, input_data, entity_placeholder_template="<entityTag={}/>"):
        """ This approach assumes that the translation won't corrupt the original
            meta-annotation for entities and objects mentioned in text.
        """

        def __optionally_register(prts):
            if len(prts) > 0:
                content.append(" ".join(prts))
            parts_to_join.clear()

        content = []
        origin_entities = []
        parts_to_join = []

        for part in input_data:
            if isinstance(part, str) and part.strip():
                parts_to_join.append(part)
            elif isinstance(part, Entity):
                entity_index = len(origin_entities)
                parts_to_join.append(entity_placeholder_template.format(entity_index))
                # Register entities information for further restoration.
                origin_entities.append(part)

        # Register original text with masked named entities.
        __optionally_register(parts_to_join)
        # Register all named entities in order of their appearance in text.
        content.extend([e.Value for e in origin_entities])

        # Due to the potential opportunity of connection lost,
        # we wrap everything in a loop with multiple attempts.
        for attempt_index in range(self.__attempts):
            try:
                # Compose text parts.
                translated_parts = [
                    part.text for part in self.translator.translate(content, dest=self.__dest, src=self.__src)
                ]

                # Take the original text.
                text = translated_parts[0]
                for entity_index in range(len(origin_entities)):
                    if entity_placeholder_template.format(entity_index) not in text:
                        return []

                # Enumerate entities.
                from_ind = 0
                text_parts = []
                for entity_index, translated_value in enumerate(translated_parts[1:]):
                    entity_placeholder_instance = entity_placeholder_template.format(entity_index)
                    # Cropping text part.
                    to_ind = text.index(entity_placeholder_instance)
                    origin_entities[entity_index].set_display_value(translated_value.strip())
                    # Register entities.
                    text_parts.append(text[from_ind:to_ind])
                    text_parts.append(origin_entities[entity_index])
                    # Update from index.
                    from_ind = to_ind + len(entity_placeholder_instance)

                # Consider the remaining part.
                text_parts.append(text[from_ind:])

                return text_parts
            except:
                if attempt_index > 0:
                    logger.info("Unable to perform translation. Try {} out of {}.".format(attempt_index, self.__attempts))
                time.sleep(self.__timeout_for_connection_lost)
                return []

    def default_pre_part_splitting_approach(self, input_data):
        """ This is the original strategy, based on the manually cropped named entities
            before the actual translation call.
        """

        def __optionally_register(prts):
            if len(prts) > 0:
                content.append(" ".join(prts))
            parts_to_join.clear()

        content = []
        origin_entities = []
        origin_entity_ind = []
        parts_to_join = []

        for _, part in enumerate(input_data):
            if isinstance(part, str) and part.strip():
                parts_to_join.append(part)
            elif isinstance(part, Entity):
                # Register first the prior parts were merged.
                __optionally_register(parts_to_join)
                # Register entities information for further restoration.
                origin_entity_ind.append(len(content))
                origin_entities.append(part)
                content.append(part.Value)

        __optionally_register(parts_to_join)

        # Due to the potential opportunity of connection lost, we wrap everything in a loop with multiple attempts.
        for attempt_index in range(self.__attempts):
            try:
                # Compose text parts.
                translated_parts = [part.text for part in
                                    self.translator.translate(content, dest=self.__dest, src=self.__src)]
                for entity_ind, entity_part_ind in enumerate(origin_entity_ind):
                    entity = origin_entities[entity_ind]
                    entity.set_display_value(translated_parts[entity_part_ind].strip())
                    translated_parts[entity_part_ind] = entity
                return translated_parts
            except:
                if attempt_index > 0:
                    logger.info("Unable to perform translation. Try {} out of {}.".format(attempt_index, self.__attempts))
                time.sleep(self.__timeout_for_connection_lost)
                return []

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        assert(isinstance(input_data, list))

        # Check the pipeline state whether is an idle mode or not.
        parent_ctx = pipeline_ctx.provide(PARENT_CTX)
        idle_mode = parent_ctx.provide(IDLE_MODE)

        # When pipeline utilized only for the assessing the expected amount
        # of rows (common case of idle_mode), there is no need to perform
        # translation.
        if idle_mode:
            return

        fast_accurate = self.fast_most_accurate_approach(input_data)
        return self.default_pre_part_splitting_approach(input_data) \
            if len(fast_accurate) == 0 else fast_accurate
