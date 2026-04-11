# ParaView Views

This note records the saved camera presets extracted from:

- `/Users/alexholland/Desktop/views.pvcvbc`

These are the raw ParaView camera values, kept in the refactor layout as Blender-facing reference data.

See also:

- `final/_blender/paraview-to-blender-info.md` for the Blender mapping rules

## Saved Views

All of the views below are perspective cameras:

- `CameraParallelProjection = 0`

### `parade`

- `CameraPosition = (-710.5998866124906, 155.04839940955577, 780.0399301944501)`
- `CameraFocalPoint = (52.83318934160216, 109.85652084191102, 56.99999999999977)`
- `CameraViewUp = (0.687286571444865, -0.010133360187298351, 0.7263156915025839)`
- `CameraViewAngle = 30`
- `CameraParallelScale = 329.5980718122458`

### `parade2`

- `CameraPosition = (-870.5486460665832, 88.7833187596505, 797.7405389891)`
- `CameraFocalPoint = (-150.080554082529, 96.80903279686298, 204.4042655918879)`
- `CameraViewUp = (0.6357066898197774, 0.0009645282511902683, 0.7719300967080076)`
- `CameraViewAngle = 28.171793383633197`
- `CameraParallelScale = 314.12815855952806`

### `City`

- `CameraPosition = (827.5660735984109, 49.13287564742965, 880.7059951185498)`
- `CameraFocalPoint = (288.24903018994945, -22.32975691174787, 333.8400148551478)`
- `CameraViewUp = (-0.7115782188004148, -0.006298457386287757, 0.702578656068758)`
- `CameraViewAngle = 28.171793383633197`
- `CameraParallelScale = 314.12815855952806`

### `uni`

Variant 1:

- `CameraPosition = (-76.76185820395062, -879.4925282617797, 863.1531793423247)`
- `CameraFocalPoint = (-13.385258127015193, 44.270460390127, 63.18321407013855)`
- `CameraViewUp = (-0.0380557896659148, 0.6556550948655048, 0.7541008907631722)`
- `CameraViewAngle = 30`
- `CameraParallelScale = 316.70257882988574`

Variant 2:

- `CameraPosition = (-65.76261356249901, -719.1700260990519, 724.3154167744245)`
- `CameraFocalPoint = (-13.385258127015193, 44.270460390127, 63.18321407013855)`
- `CameraViewUp = (-0.0380557896659148, 0.6556550948655048, 0.7541008907631722)`
- `CameraViewAngle = 30`
- `CameraParallelScale = 316.70257882988574`

## Blender Mapping Reminder

For the current Blender matching work:

- set location from `CameraPosition`
- aim toward `CameraFocalPoint`
- use `CameraViewUp` to resolve roll
- treat `CameraViewAngle` as vertical FOV by default
- set Blender perspective cameras with `sensor_fit = 'VERTICAL'` and `angle_y = radians(CameraViewAngle)`

Exact framing still depends on the ParaView viewport aspect ratio or a reference screenshot.
