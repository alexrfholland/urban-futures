ENABLE_CLIPBOX_SETUP = False
ENABLE_CAMERA_CLIPBOXES = False


def any_clipbox_scripts_enabled() -> bool:
    return ENABLE_CLIPBOX_SETUP or ENABLE_CAMERA_CLIPBOXES
