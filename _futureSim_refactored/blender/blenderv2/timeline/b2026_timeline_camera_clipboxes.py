import bpy
from bpy.app.handlers import persistent
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import b2026_timeline_scene_contract as scene_contract
import b2026_timeline_runtime_flags as runtime_flags


SCENE_SPECS = {
    "city": {
        "camera_collection": scene_contract.get_collection_name("city", "cameras", legacy=True),
        "live_clip_object": scene_contract.SITE_CONTRACTS["city"]["live_clip_object"],
        "proxy_prefix": scene_contract.SITE_CONTRACTS["city"]["camera_proxy_prefix"],
        "subcollection_prefix": scene_contract.SITE_CONTRACTS["city"]["camera_subcollection_prefix"],
    },
    "parade": {
        "camera_collection": scene_contract.get_collection_name("trimmed-parade", "cameras", legacy=True),
        "live_clip_object": scene_contract.SITE_CONTRACTS["trimmed-parade"]["live_clip_object"],
        "proxy_prefix": scene_contract.SITE_CONTRACTS["trimmed-parade"]["camera_proxy_prefix"],
        "subcollection_prefix": scene_contract.SITE_CONTRACTS["trimmed-parade"]["camera_subcollection_prefix"],
    },
    "uni": {
        "camera_collection": scene_contract.get_collection_name("uni", "cameras", legacy=True),
        "live_clip_object": scene_contract.SITE_CONTRACTS["uni"]["live_clip_object"],
        "proxy_prefix": scene_contract.SITE_CONTRACTS["uni"]["camera_proxy_prefix"],
        "subcollection_prefix": scene_contract.SITE_CONTRACTS["uni"]["camera_subcollection_prefix"],
    },
}

ENABLE_CAMERA_CLIPBOXES = runtime_flags.ENABLE_CAMERA_CLIPBOXES
HANDLER_NAMES = {
    "load_post": "camera_clipbox_load_post",
    "depsgraph_update_post": "camera_clipbox_depsgraph_post",
    "render_pre": "camera_clipbox_render_pre",
}

_LAST_SYNC_STATE = {}


def iter_cameras_in_collection(collection):
    seen = set()
    for obj in list(collection.all_objects):
        if obj.type != "CAMERA":
            continue
        if obj.name in seen:
            continue
        seen.add(obj.name)
        yield obj


def find_direct_parent_collection(parent_collection, obj):
    for child in parent_collection.children:
        if obj.name in child.objects:
            return child
    return None


def ensure_subcollection(parent_collection, camera, prefix):
    name = f"{prefix}{camera.name}"
    sub = bpy.data.collections.get(name)
    if sub is None:
        sub = bpy.data.collections.new(name)
    if sub.name not in parent_collection.children:
        parent_collection.children.link(sub)
    if camera.name not in sub.objects:
        sub.objects.link(camera)
    if camera.name in parent_collection.objects:
        parent_collection.objects.unlink(camera)
    return sub


def ensure_proxy_clip_object(live_clip, subcollection, proxy_name):
    proxy = bpy.data.objects.get(proxy_name)
    if proxy is None:
        mesh = live_clip.data.copy()
        proxy = bpy.data.objects.new(proxy_name, mesh)
        proxy.matrix_world = live_clip.matrix_world.copy()
        proxy.scale = live_clip.scale.copy()
        proxy.rotation_euler = live_clip.rotation_euler.copy()
        proxy.location = live_clip.location.copy()
        proxy.hide_render = True
        proxy.display_type = "WIRE"
    if proxy.name not in subcollection.objects:
        subcollection.objects.link(proxy)
    for collection in tuple(proxy.users_collection):
        if collection != subcollection and proxy.name in collection.objects:
            collection.objects.unlink(proxy)
    return proxy


def copy_clip_transform(source, target):
    target.location = source.location.copy()
    if hasattr(source, "rotation_euler") and hasattr(target, "rotation_euler"):
        target.rotation_euler = source.rotation_euler.copy()
    target.scale = source.scale.copy()


def ensure_camera_clipboxes():
    if not ENABLE_CAMERA_CLIPBOXES:
        return

    for scene_name, spec in SCENE_SPECS.items():
        scene = bpy.data.scenes.get(scene_name)
        camera_collection = bpy.data.collections.get(spec["camera_collection"])
        live_clip = bpy.data.objects.get(spec["live_clip_object"])
        if scene is None or camera_collection is None or live_clip is None:
            continue

        for camera in iter_cameras_in_collection(camera_collection):
            sub = ensure_subcollection(
                camera_collection, camera, spec["subcollection_prefix"]
            )
            proxy_name = f"{spec['proxy_prefix']}{camera.name}"
            proxy = ensure_proxy_clip_object(live_clip, sub, proxy_name)
            camera["clip_proxy_object"] = proxy.name


def stamp_camera_proxy_from_live_clip(scene, camera=None):
    if not ENABLE_CAMERA_CLIPBOXES:
        return None

    spec = SCENE_SPECS.get(scene.name)
    if spec is None:
        return None

    live_clip = bpy.data.objects.get(spec["live_clip_object"])
    target_camera = camera or scene.camera
    if live_clip is None or target_camera is None:
        return None

    proxy_name = target_camera.get("clip_proxy_object")
    if not proxy_name:
        return None

    proxy = bpy.data.objects.get(proxy_name)
    if proxy is None:
        return None

    copy_clip_transform(live_clip, proxy)
    return proxy


def sync_scene_clipbox(scene):
    if not ENABLE_CAMERA_CLIPBOXES:
        return

    spec = SCENE_SPECS.get(scene.name)
    if spec is None or scene.camera is None:
        return
    live_clip = bpy.data.objects.get(spec["live_clip_object"])
    if live_clip is None:
        return
    proxy_name = scene.camera.get("clip_proxy_object")
    if not proxy_name:
        return
    proxy = bpy.data.objects.get(proxy_name)
    if proxy is None:
        return
    state_key = scene.name
    new_state = (
        scene.camera.name,
        proxy_name,
        tuple(proxy.location),
        tuple(proxy.scale),
        tuple(proxy.rotation_euler) if hasattr(proxy, "rotation_euler") else None,
    )
    if _LAST_SYNC_STATE.get(state_key) == new_state:
        return
    live_clip.location = proxy.location.copy()
    if hasattr(live_clip, "rotation_euler") and hasattr(proxy, "rotation_euler"):
        live_clip.rotation_euler = proxy.rotation_euler.copy()
    live_clip.scale = proxy.scale.copy()
    live_clip.hide_render = True
    live_clip.hide_viewport = False
    _LAST_SYNC_STATE[state_key] = new_state


def sync_all_scene_clipboxes():
    if not ENABLE_CAMERA_CLIPBOXES:
        return

    for scene_name in SCENE_SPECS:
        scene = bpy.data.scenes.get(scene_name)
        if scene is not None:
            sync_scene_clipbox(scene)


@persistent
def camera_clipbox_load_post(_dummy):
    ensure_camera_clipboxes()
    sync_all_scene_clipboxes()


@persistent
def camera_clipbox_depsgraph_post(_dummy):
    sync_all_scene_clipboxes()


@persistent
def camera_clipbox_render_pre(scene, _dummy):
    sync_scene_clipbox(scene)


def remove_handlers_by_name(handler_list, function_name):
    for handler in list(handler_list):
        if getattr(handler, "__name__", "") == function_name:
            handler_list.remove(handler)


def register():
    if not ENABLE_CAMERA_CLIPBOXES:
        _LAST_SYNC_STATE.clear()
        return

    ensure_camera_clipboxes()
    sync_all_scene_clipboxes()
    remove_handlers_by_name(bpy.app.handlers.load_post, HANDLER_NAMES["load_post"])
    remove_handlers_by_name(
        bpy.app.handlers.depsgraph_update_post,
        HANDLER_NAMES["depsgraph_update_post"],
    )
    remove_handlers_by_name(bpy.app.handlers.render_pre, HANDLER_NAMES["render_pre"])
    bpy.app.handlers.load_post.append(camera_clipbox_load_post)
    bpy.app.handlers.depsgraph_update_post.append(camera_clipbox_depsgraph_post)
    bpy.app.handlers.render_pre.append(camera_clipbox_render_pre)


def unregister():
    remove_handlers_by_name(bpy.app.handlers.load_post, HANDLER_NAMES["load_post"])
    remove_handlers_by_name(
        bpy.app.handlers.depsgraph_update_post,
        HANDLER_NAMES["depsgraph_update_post"],
    )
    remove_handlers_by_name(bpy.app.handlers.render_pre, HANDLER_NAMES["render_pre"])
    _LAST_SYNC_STATE.clear()


if __name__ == "__main__":
    register()
