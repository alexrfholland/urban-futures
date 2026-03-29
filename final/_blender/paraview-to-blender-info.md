# ParaView Camera To Blender

This note captures the minimum camera data needed to move a saved ParaView view into Blender, and the repo-specific steps required to make that camera work in the heavy 2026 blends.

## What A ParaView Camera Save Gives You

From the current `.pvcvbc` camera save format, the useful fields are:

- `CameraPosition`
- `CameraFocalPoint`
- `CameraViewUp`
- `CameraViewAngle`
- `CameraParallelProjection`
- `CameraParallelScale`

These are enough to reconstruct:

- camera location
- camera direction / orientation
- perspective vs orthographic mode
- one FOV value for perspective cameras
- ortho scale for parallel cameras

## What It Does Not Give You

The camera save does not include:

- lens length in mm
- sensor size
- sensor fit
- viewport width / height
- render aspect ratio
- pixel aspect ratio
- Blender clip-box / crop volumes

That means a ParaView camera save is enough to get the angle and orientation right, but not always enough to get exact framing parity without one more piece of information: the ParaView viewport aspect ratio or a reference screenshot.

## The Correct Blender Mapping

Use these mappings when importing a ParaView camera into Blender:

1. Set Blender camera location from `CameraPosition`.
2. Aim the camera from `CameraPosition` toward `CameraFocalPoint`.
3. Use `CameraViewUp` to resolve roll.
4. If `CameraParallelProjection == 1`, use an orthographic camera and map `CameraParallelScale` to Blender orthographic scale.
5. If `CameraParallelProjection == 0`, use a perspective camera.
6. Treat `CameraViewAngle` as a vertical FOV by default.
7. In Blender, set:
   - `camera.data.type = 'PERSP'`
   - `camera.data.sensor_fit = 'VERTICAL'`
   - `camera.data.angle_y = radians(CameraViewAngle)`

For this project, step 7 is the important one. A generic Blender `camera.data.angle` assignment can preserve the view direction but still give a shot that is too tight or too loose.

## Why The Framing Can Still Be Wrong

Even with the correct position, focal point, up vector, and vertical FOV, exact framing can still differ because:

- ParaView viewport aspect ratio is not stored in the camera save.
- Blender render aspect ratio may not match the ParaView window.
- The heavy Blender blends use extra clip-box culling that does not exist in the ParaView file.

In practice:

- if the shot is the right angle but too zoomed in or out, the missing piece is usually aspect handling
- if the shot is black or visibly cut by scene culling, the missing piece is usually the Blender clip-box setup, not the ParaView camera data

## Repo-Specific Requirement: Clip Boxes

For the 2026 heavy blends, importing the camera is not enough.

After placing or updating a camera in:

- `City_Camera`
- `Parade_Camera`

rerun these scripts in order:

1. `final/_blender/2026/b2026_clipbox_setup.py`
2. `final/_blender/2026/b2026_camera_clipboxes.py`

This is required because the renderable crop in these scenes is controlled by Blender clip boxes and camera proxy clip boxes, not by the ParaView file.

## If You Need Lens Length Anyway

If you want a photographic equivalent lens length, you can derive it after choosing a sensor dimension:

`lens_mm = sensor_dim_mm / (2 * tan(FOV / 2))`

Use:

- sensor height if `CameraViewAngle` is treated as vertical FOV
- sensor width if it is treated as horizontal FOV

For this repo, matching `angle_y` is the safer primary target. Lens length is secondary.

## Minimum Checklist

To make a ParaView camera work in Blender, collect or infer:

- `CameraPosition`
- `CameraFocalPoint`
- `CameraViewUp`
- `CameraViewAngle`
- `CameraParallelProjection`
- `CameraParallelScale` if orthographic
- ParaView viewport aspect ratio or a screenshot, if exact framing matters
- Blender clip-box refresh, if working in the heavy 2026 scene files
