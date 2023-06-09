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

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        assert(isinstance(input_data, list))

        def __optionally_register(prts_to_join):
            if len(prts_to_join) > 0:
                content.append(" ".join(prts_to_join))
            parts_to_join.clear()

        # Check the pipeline state whether is an idle mode or not.
        parent_ctx = pipeline_ctx.provide(PARENT_CTX)
        idle_mode = parent_ctx.provide(IDLE_MODE)

        # When pipeline utilized only for the assessing the expected amount
        # of rows (common case of idle_mode), there is no need to perform
        # translation.
        if idle_mode:
            return

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

        translated_parts = []

        # Due to the potential opportunity of connection lost, we wrap everything in a loop with multiple attempts.
        for attempt_index in range(self.__attempts):
            try:
                # Compose text parts.
                translated_parts = [part.text for part in
                                    self.translator.translate(content, dest=self.__dest, src=self.__src)]
                for entity_ind, entity_part_ind in enumerate(origin_entity_ind):
                    entity = origin_entities[entity_ind]
                    entity.set_display_value(translated_parts[entity_part_ind])
                    translated_parts[entity_part_ind] = entity
                break
            except:
                logger.info("Unable to perform translation. Try {} out of {}.".format(attempt_index, self.__attempts))
                time.sleep(self.__timeout_for_connection_lost)
                translated_parts = []

        return translated_parts
