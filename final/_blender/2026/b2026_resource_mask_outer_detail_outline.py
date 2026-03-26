import bpy


FRAME_NAME = "ResourceMaskOuterDetailOutline::Frame"
FRAME_LABEL = "Resource Mask Outer Detail Outline"
NODE_PREFIX = "ResourceMaskOuterDetailOutline::"

FRAME_LOCATION = (14500.0, 1200.0)
OUTLINE_COLOR = (0.96, 0.70, 0.54, 1.0)

# Detailed edge branch
DETAIL_BLUR_PIXELS = 1
DETAIL_EDGE_LOW = 0.08
DETAIL_EDGE_HIGH = 0.16

# Coarse silhouette branch
OUTER_CLOSE_GROW = 6
OUTER_BLUR_PIXELS = 5
OUTER_CLOSE_SHRINK = -4
OUTER_BAND_GROW = 4
OUTER_BAND_SHRINK = -4
OUTER_BAND_LOW = 0.01
OUTER_BAND_HIGH = 0.01


def require_active_scene():
    scene = bpy.context.scene
    if scene is None or not scene.use_nodes or scene.node_tree is None:
        raise ValueError("Active scene does not have compositor nodes enabled.")
    return scene


def get_input(node, *names, index=0):
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            return socket
    return node.inputs[index]


def get_output(node, *names, index=0):
    for name in names:
        socket = node.outputs.get(name)
        if socket is not None:
            return socket
    return node.outputs[index]


def ensure_link(node_tree, from_socket, to_socket):
    for link in list(to_socket.links):
        if link.from_socket == from_socket:
            return
        node_tree.links.remove(link)
    node_tree.links.new(from_socket, to_socket)


def ensure_frame(node_tree):
    frame = node_tree.nodes.get(FRAME_NAME)
    if frame is None or frame.bl_idname != "NodeFrame":
        frame = node_tree.nodes.new("NodeFrame")
    frame.name = FRAME_NAME
    frame.label = FRAME_LABEL
    frame.location = FRAME_LOCATION
    frame.use_custom_color = True
    frame.color = (0.14, 0.14, 0.14)
    frame.shrink = True
    return frame


def ensure_node(node_tree, bl_idname, name, label, location, parent):
    node = node_tree.nodes.get(name)
    if node is None or node.bl_idname != bl_idname:
        if node is not None:
            node_tree.nodes.remove(node)
        node = node_tree.nodes.new(bl_idname)
    node.name = name
    node.label = label
    node.location = location
    node.parent = parent
    return node


def main():
    scene = require_active_scene()
    node_tree = scene.node_tree
    frame = ensure_frame(node_tree)

    mask_input = ensure_node(
        node_tree,
        "NodeReroute",
        f"{NODE_PREFIX}Input",
        "resource_mask_binary",
        (0.0, 0.0),
        frame,
    )

    # Detailed edge branch: preserve outer branch detail.
    detail_blur = ensure_node(
        node_tree,
        "CompositorNodeBlur",
        f"{NODE_PREFIX}DetailBlur",
        "Detail blur",
        (220.0, 260.0),
        frame,
    )
    detail_blur.filter_type = "GAUSS"
    detail_blur.use_relative = False
    detail_blur.size_x = DETAIL_BLUR_PIXELS
    detail_blur.size_y = DETAIL_BLUR_PIXELS

    detail_edge = ensure_node(
        node_tree,
        "CompositorNodeFilter",
        f"{NODE_PREFIX}DetailEdge",
        "Detailed edge detect",
        (480.0, 260.0),
        frame,
    )
    detail_edge.filter_type = "KIRSCH"

    detail_ramp = ensure_node(
        node_tree,
        "CompositorNodeValToRGB",
        f"{NODE_PREFIX}DetailRamp",
        "Detailed edge threshold",
        (740.0, 260.0),
        frame,
    )
    detail_ramp.color_ramp.interpolation = "LINEAR"
    detail_ramp.color_ramp.elements[0].position = DETAIL_EDGE_LOW
    detail_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    detail_ramp.color_ramp.elements[1].position = DETAIL_EDGE_HIGH
    detail_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    detail_output = ensure_node(
        node_tree,
        "NodeReroute",
        f"{NODE_PREFIX}DetailEdges",
        "resource_detail_edges",
        (1020.0, 360.0),
        frame,
    )

    # Outer band branch: create a coarse boundary zone and ignore interior gaps.
    close_grow = ensure_node(
        node_tree,
        "CompositorNodeDilateErode",
        f"{NODE_PREFIX}CloseGrow",
        "Close gaps (grow)",
        (220.0, -80.0),
        frame,
    )
    close_grow.mode = "DISTANCE"
    close_grow.distance = OUTER_CLOSE_GROW

    coarse_blur = ensure_node(
        node_tree,
        "CompositorNodeBlur",
        f"{NODE_PREFIX}CoarseBlur",
        "Coarse blur",
        (480.0, -80.0),
        frame,
    )
    coarse_blur.filter_type = "GAUSS"
    coarse_blur.use_relative = False
    coarse_blur.size_x = OUTER_BLUR_PIXELS
    coarse_blur.size_y = OUTER_BLUR_PIXELS

    coarse_threshold = ensure_node(
        node_tree,
        "CompositorNodeValToRGB",
        f"{NODE_PREFIX}CoarseThreshold",
        "Coarse threshold",
        (740.0, -80.0),
        frame,
    )
    coarse_threshold.color_ramp.interpolation = "CONSTANT"
    coarse_threshold.color_ramp.elements[0].position = 0.5
    coarse_threshold.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    coarse_threshold.color_ramp.elements[1].position = 0.5
    coarse_threshold.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    close_shrink = ensure_node(
        node_tree,
        "CompositorNodeDilateErode",
        f"{NODE_PREFIX}CloseShrink",
        "Restore size",
        (1000.0, -80.0),
        frame,
    )
    close_shrink.mode = "DISTANCE"
    close_shrink.distance = OUTER_CLOSE_SHRINK

    outer_band_grow = ensure_node(
        node_tree,
        "CompositorNodeDilateErode",
        f"{NODE_PREFIX}OuterBandGrow",
        "Outer band grow",
        (1260.0, 20.0),
        frame,
    )
    outer_band_grow.mode = "DISTANCE"
    outer_band_grow.distance = OUTER_BAND_GROW

    outer_band_shrink = ensure_node(
        node_tree,
        "CompositorNodeDilateErode",
        f"{NODE_PREFIX}OuterBandShrink",
        "Outer band shrink",
        (1260.0, -180.0),
        frame,
    )
    outer_band_shrink.mode = "DISTANCE"
    outer_band_shrink.distance = OUTER_BAND_SHRINK

    outer_band_subtract = ensure_node(
        node_tree,
        "CompositorNodeMath",
        f"{NODE_PREFIX}OuterBandSubtract",
        "Outer band",
        (1520.0, -80.0),
        frame,
    )
    outer_band_subtract.operation = "SUBTRACT"
    outer_band_subtract.use_clamp = True

    outer_band_ramp = ensure_node(
        node_tree,
        "CompositorNodeValToRGB",
        f"{NODE_PREFIX}OuterBandRamp",
        "Outer band threshold",
        (1780.0, -80.0),
        frame,
    )
    outer_band_ramp.color_ramp.interpolation = "CONSTANT"
    outer_band_ramp.color_ramp.elements[0].position = OUTER_BAND_LOW
    outer_band_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    outer_band_ramp.color_ramp.elements[1].position = OUTER_BAND_HIGH
    outer_band_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    coarse_output = ensure_node(
        node_tree,
        "NodeReroute",
        f"{NODE_PREFIX}CoarseMask",
        "resource_coarse_mask",
        (1260.0, -320.0),
        frame,
    )
    band_output = ensure_node(
        node_tree,
        "NodeReroute",
        f"{NODE_PREFIX}OuterBand",
        "resource_outer_band",
        (2040.0, -80.0),
        frame,
    )

    # Gate fine edges by the outer band.
    multiply = ensure_node(
        node_tree,
        "CompositorNodeMixRGB",
        f"{NODE_PREFIX}Multiply",
        "Detail edges x outer band",
        (2040.0, 200.0),
        frame,
    )
    multiply.blend_type = "MULTIPLY"
    multiply.use_alpha = False
    get_input(multiply, "Fac", index=0).default_value = 1.0

    gated_output = ensure_node(
        node_tree,
        "NodeReroute",
        f"{NODE_PREFIX}GatedMask",
        "resource_edge_mask_outer_detail",
        (2300.0, 280.0),
        frame,
    )

    color = ensure_node(
        node_tree,
        "CompositorNodeRGB",
        f"{NODE_PREFIX}Color",
        "Outline colour",
        (2300.0, 40.0),
        frame,
    )
    get_output(color, "Image").default_value = OUTLINE_COLOR

    set_alpha = ensure_node(
        node_tree,
        "CompositorNodeSetAlpha",
        f"{NODE_PREFIX}SetAlpha",
        "Coloured outer detail outline",
        (2560.0, 180.0),
        frame,
    )
    set_alpha.mode = "APPLY"

    outline_output = ensure_node(
        node_tree,
        "NodeReroute",
        f"{NODE_PREFIX}Outline",
        "resource_outline_outer_detail",
        (2820.0, 180.0),
        frame,
    )

    ensure_link(node_tree, get_output(mask_input, index=0), get_input(detail_blur, "Image"))
    ensure_link(node_tree, get_output(detail_blur, "Image"), get_input(detail_edge, "Image"))
    ensure_link(node_tree, get_output(detail_edge, "Image"), get_input(detail_ramp, "Fac"))
    ensure_link(node_tree, get_output(detail_ramp, "Image"), get_input(detail_output, index=0))

    ensure_link(node_tree, get_output(mask_input, index=0), get_input(close_grow, "Mask"))
    ensure_link(node_tree, get_output(close_grow, "Mask"), get_input(coarse_blur, "Image"))
    ensure_link(node_tree, get_output(coarse_blur, "Image"), get_input(coarse_threshold, "Fac"))
    ensure_link(node_tree, get_output(coarse_threshold, "Image"), get_input(close_shrink, "Mask"))
    ensure_link(node_tree, get_output(close_shrink, "Mask"), get_input(coarse_output, index=0))
    ensure_link(node_tree, get_output(close_shrink, "Mask"), get_input(outer_band_grow, "Mask"))
    ensure_link(node_tree, get_output(close_shrink, "Mask"), get_input(outer_band_shrink, "Mask"))
    ensure_link(node_tree, get_output(outer_band_grow, "Mask"), get_input(outer_band_subtract, index=0))
    ensure_link(node_tree, get_output(outer_band_shrink, "Mask"), get_input(outer_band_subtract, index=1))
    ensure_link(node_tree, get_output(outer_band_subtract, "Value"), get_input(outer_band_ramp, "Fac"))
    ensure_link(node_tree, get_output(outer_band_ramp, "Image"), get_input(band_output, index=0))

    ensure_link(node_tree, get_output(detail_ramp, "Image"), get_input(multiply, index=1))
    ensure_link(node_tree, get_output(outer_band_ramp, "Image"), get_input(multiply, index=2))
    ensure_link(node_tree, get_output(multiply, "Image"), get_input(gated_output, index=0))

    ensure_link(node_tree, get_output(color, "Image"), get_input(set_alpha, "Image", index=0))
    ensure_link(node_tree, get_output(multiply, "Image"), get_input(set_alpha, "Alpha", index=1))
    ensure_link(node_tree, get_output(set_alpha, "Image", index=0), get_input(outline_output, index=0))

    print(f"Built '{FRAME_LABEL}' in scene '{scene.name}'.")
    print("Connect your binary mask to 'resource_mask_binary'.")
    print("Useful outputs:")
    print("- resource_detail_edges")
    print("- resource_coarse_mask")
    print("- resource_outer_band")
    print("- resource_edge_mask_outer_detail")
    print("- resource_outline_outer_detail")


if __name__ == "__main__":
    main()
