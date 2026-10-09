# kaggle_v2 image-level checks. Exploratory, not pre-registered

Every figure is a twelve-cell mean AUROC unless it says otherwise, with a 95 percent split-block interval, 10,000 draws, the twelve cells averaged inside each draw. These checks were chosen after the data had been seen. They are leads, not results.

## The filename carries the label

kaggle_v2 aggregates several collections, and the collection predicts the label. A score that looks at no pixel, the positive share of the image's filename family among the other blocks, reaches an AUROC of 0.811 [0.681, 0.940] on 72 positive and 69 negative test images. The scores are computed once on the full split. The interval resamples blocks under them.

| Filename family | Test images | Positive | Blocks |
|---|---|---|---|
| `SampleV1_2_mp4` | 43 | 31 | 1 |
| `bad_weld_vid` | 25 | 23 | 5 |
| `Screenshot` | 14 | 6 | 13 |
| `good_weld_vid` | 14 | 0 | 10 |
| `good-tig-welds` | 12 | 0 | 12 |
| `good-stick-welds` | 9 | 2 | 9 |
| `mig-good-welds` | 5 | 0 | 5 |
| `Spatters-welding-images` | 4 | 3 | 4 |
| `Good-Welding-images` | 3 | 0 | 3 |
| `Slag-inclusion-welding-images` | 3 | 1 | 3 |
| `carbon-steel-good-welds` | 3 | 0 | 3 |

Families with fewer than three test images are left out of this list and kept in the score.

## The label rule, and the video block left out

The split holds 72 images with a `Defect` box, 11 with a `Bad Weld` box and no `Defect` box, and 58 with neither. For "Bad Weld or Defect present" the in-domain score is the strongest `Bad Weld` or `Defect` prediction, which is what `KAGGLE_BAD_WELD_IS_DEFECT = True` would give. Foreign models keep their frozen score. The video block, kaggle_v2:block:000043, holds the `SampleV1_2_mp4` frames. Without it the split holds 41 positive and 57 negative images in 73 blocks.

| Trained on | Frozen rule | Bad Weld or Defect present | Defect against Good Weld only | Frozen rule, video block left out |
|---|---|---|---|---|
| weld_surface_v2 | 0.555 [0.448, 0.618] | 0.496 [0.389, 0.556] | 0.525 [0.413, 0.586] | 0.536 [0.418, 0.627] |
| kaggle_v2, in domain | 0.764 [0.692, 0.843] | 0.828 [0.764, 0.910] | 0.794 [0.731, 0.873] | 0.770 [0.679, 0.861] |
| lohi_weld | 0.499 [0.430, 0.624] | 0.524 [0.458, 0.636] | 0.514 [0.441, 0.639] | 0.567 [0.485, 0.641] |

## The score rule

The frozen score is the strongest defect-class prediction. The alternatives below use the same predictions, before NMS, restricted to the model's own defect classes.

| Score rule | Trained on | Over four | yolo11n | yolo11s | rtdetr_l | faster_rcnn |
|---|---|---|---|---|---|---|
| sum of the scores | weld_surface_v2 | 0.572 [0.451, 0.631] | 0.593 [0.424, 0.679] | 0.576 [0.418, 0.675] | 0.558 [0.412, 0.629] | 0.562 [0.454, 0.678] |
| sum of the scores | kaggle_v2 | 0.757 [0.676, 0.835] | 0.753 [0.616, 0.857] | 0.779 [0.683, 0.883] | 0.744 [0.651, 0.837] | 0.752 [0.672, 0.826] |
| sum of the scores | lohi_weld | 0.533 [0.466, 0.655] | 0.534 [0.453, 0.666] | 0.527 [0.444, 0.652] | 0.468 [0.366, 0.670] | 0.602 [0.507, 0.701] |
| count of predictions at or above 0.25 | weld_surface_v2 | 0.557 [0.468, 0.608] | 0.566 [0.438, 0.630] | 0.550 [0.435, 0.624] | 0.558 [0.411, 0.631] | 0.556 [0.475, 0.670] |
| count of predictions at or above 0.25 | kaggle_v2 | 0.650 [0.582, 0.731] | 0.625 [0.541, 0.722] | 0.668 [0.561, 0.784] | 0.684 [0.613, 0.779] | 0.623 [0.570, 0.680] |
| count of predictions at or above 0.25 | lohi_weld | 0.535 [0.472, 0.658] | 0.534 [0.462, 0.678] | 0.508 [0.433, 0.632] | 0.477 [0.368, 0.689] | 0.623 [0.528, 0.714] |
| count of predictions at or above 0.10 | weld_surface_v2 | 0.559 [0.453, 0.615] | 0.570 [0.437, 0.652] | 0.548 [0.425, 0.644] | 0.570 [0.410, 0.638] | 0.546 [0.456, 0.653] |
| count of predictions at or above 0.10 | kaggle_v2 | 0.698 [0.629, 0.770] | 0.672 [0.553, 0.761] | 0.740 [0.639, 0.849] | 0.710 [0.626, 0.792] | 0.670 [0.605, 0.747] |
| count of predictions at or above 0.10 | lohi_weld | 0.531 [0.465, 0.646] | 0.527 [0.447, 0.656] | 0.527 [0.444, 0.650] | 0.460 [0.363, 0.647] | 0.610 [0.503, 0.708] |
| mean of the top three, zero where fewer | weld_surface_v2 | 0.572 [0.452, 0.635] | 0.582 [0.418, 0.668] | 0.568 [0.418, 0.662] | 0.585 [0.418, 0.666] | 0.553 [0.449, 0.672] |
| mean of the top three, zero where fewer | kaggle_v2 | 0.766 [0.692, 0.844] | 0.749 [0.619, 0.853] | 0.790 [0.696, 0.895] | 0.772 [0.699, 0.862] | 0.752 [0.674, 0.826] |
| mean of the top three, zero where fewer | lohi_weld | 0.518 [0.444, 0.658] | 0.541 [0.456, 0.699] | 0.506 [0.427, 0.663] | 0.469 [0.348, 0.690] | 0.554 [0.468, 0.657] |

## One lohi_weld class at a time

| Class | Over four | yolo11n | yolo11s | rtdetr_l | faster_rcnn |
|---|---|---|---|---|---|
| pore | 0.554 [0.478, 0.716] | 0.552 [0.472, 0.735] | 0.540 [0.471, 0.675] | 0.517 [0.397, 0.776] | 0.605 [0.521, 0.739] |
| deposit | 0.615 [0.534, 0.686] | 0.686 [0.566, 0.761] | 0.664 [0.570, 0.728] | 0.478 [0.379, 0.633] | 0.630 [0.502, 0.713] |
| discontinuity | 0.482 [0.398, 0.615] | 0.470 [0.374, 0.615] | 0.449 [0.355, 0.554] | 0.531 [0.414, 0.728] | 0.476 [0.388, 0.619] |
| stain | 0.484 [0.413, 0.593] | 0.475 [0.376, 0.633] | 0.456 [0.375, 0.589] | 0.447 [0.354, 0.605] | 0.559 [0.448, 0.644] |

Over the twelve lohi_weld cells, the class supplying the strongest prediction on a defect-free kaggle_v2 image: stain 329 of 828 (40 percent), discontinuity 227 of 828 (27 percent), pore 145 of 828 (18 percent), deposit 127 of 828 (15 percent).
