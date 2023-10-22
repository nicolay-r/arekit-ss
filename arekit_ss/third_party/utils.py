from arekit_ss.utils import auto_import


def create_translate_model(backend):

    if backend == "googletrans":
        # We do auto-import so we not depend on the actually installed library.
        translate_value = auto_import("arekit_ss.third_party.googletrans.translate_value")
        # Translation of the list of data.
        # Returns the list of strings.
        return lambda str_list, src, dest: [translate_value(s, dest=dest, src=src) for s in str_list]
