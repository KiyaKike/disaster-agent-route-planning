---
license: cc-by-nc-sa-4.0
pretty_name: "xBD / xView2 - building damage assessment"
size_categories:
- 10K<n<100K
task_categories:
- image-segmentation
tags:
- earth-observation
- remote-sensing
- taco
---

<p align="center"><img src="assets/logos.png" width="546" alt="Image and Signal Processing Group, Universitat de València · ELLIOT"></p>

# xBD / xView2 - building damage assessment

> **This is a repackaging, not a new dataset.** It is **xBD / xView2 - building damage assessment** by DIU / CMU SEI, Maxar Open Data (Gupta et al.), converted to TACO. Pixel values and labels are kept as released except where the description below says otherwise. All credit belongs to the original authors: if you use it, please cite them and follow their licence.
>
> [original dataset](https://xview2.org/) · [paper](https://doi.org/10.48550/arXiv.1911.09296) · licence: CC-BY-NC-SA-4.0
>
> Repackaged into TACO by the [Image and Signal Processing Group (ISP)](https://isp.uv.es/), Universitat de València, within the [ELLIOT](https://elliot-ai.eu) project.

## Citation

Please cite the original work:

```bibtex
@article{gupta2019xbd,
  title   = {xBD: A Dataset for Assessing Building Damage from Satellite Imagery},
  author  = {Gupta, Ritwik and Hosfelt, Richard and Sajeev, Sandra and Patel, Nirav and Goodman, Bryce and Doshi, Jigar and Heim, Eric and Choset, Howie and Gaston, Matthew},
  journal = {arXiv preprint arXiv:1911.09296},
  year    = {2019}
}
```

## About the data

11034 pre/post-disaster WorldView pairs of 1024x1024 over 19 events, every building outlined and graded on the release's four-level damage scale plus an `un-classified` grade that is the ignore index. Six hazard types: fire, flooding, wind, earthquake, tsunami and volcano.

**11,034 samples** · splits: `test` 1,866 · `train` 7,467 · `validation` 1,701 · tasks: `instance-segmentation`, `change-detection`

Packaged as [TACO v3](https://asterisk.coop/taco/spec/).

<details>
<summary>Full description</summary>

**Targets.** Outlines come from the post label, where the grade lives, as flat pixel rings with a per-object vertex count. The release's WGS84 ring is preserved beside them because the PNGs carry no geotransform.

**Capture geometry.** Kept per sample for both halves -- sensor, GSD, off-nadir angle, sun elevation -- since the two passes differ and that difference is a confound.

**Splits.**
- tier-3 is more training data rather than a third split, and is marked `release_tier` (train 2799, tier3 6369, hold 933, test 933).
- `hold` is the challenge holdout, distributed separately by xview2.org and labelled like train and tier3; it and xBD's labelled `test` archive form this container's `test` (933 + 933), with the archive name in `split_original`.
- `val` is carved from train + tier3 on whole events.
- The challenge's unlabelled half is `xview2_test`.

**Scope.** xBD's whole named release. The 933 pairs of the anonymised challenge test archive, which carry no disaster and no scene id, are in the held-out half below. PNGs are stored byte-for-byte.

</details>

## Held-out half

`xview2_test.zip` holds the 933 samples whose targets the publisher withheld.
Same inputs, no target slots: it is there to be predicted on and submitted, and
it belongs in no training mixture.

```python
import os
from huggingface_hub import snapshot_download
from taco.ml import Dataset

root = snapshot_download("isp-uv-es/xview2-taco", repo_type="dataset",
                         allow_patterns=["xview2_test.zip", "xview2_test.zip/*", "xview2_test.zip/.tacocat/*"])
ds = Dataset(os.path.join(root, "xview2_test.zip"))
```

## Getting started

```bash
git clone --recursive https://github.com/OscarPellicer/taco
pip install -e "taco/python[ml]"     # builds the reader: C++23, CMake, Ninja, pkg-config, libcurl >= 7.83, OpenSSL >= 3
```

Read it straight from the Hub:

```python
import os
from huggingface_hub import hf_hub_download, snapshot_download
from taco.ml import Dataset, plot_sample

path = hf_hub_download("isp-uv-es/xview2-taco", "xview2.zip", repo_type="dataset")
ds = Dataset(path)
plot_sample(ds[0])
```

or from a local copy:

```python
ds = Dataset("xview2.zip")
sample = ds[0]                      # {slot name: SlotValue}, arrays decoded
sample["pre"].array.shape
```

Metadata without decoding anything:

```python
import taco
taco.read("xview2.zip")              # one Arrow table, levels joined
```

## Samples

![sample](assets/sample_0.png)
![sample](assets/sample_1.png)
![sample](assets/sample_2.png)
![sample](assets/sample_3.png)
![sample](assets/sample_4.png)

## What a sample contains

| role | slot | holds | modality | detail |
|---|---|---|---|---|
| input | `pre` | raster | optical | 3 band(s), requantised |
| input | `post` | raster | optical | 3 band(s), requantised |
| target | `buildings` | polygon |  |  |
| target | `damage` | class_sequence |  | 5 classes |

## Licence

CC-BY-NC-SA-4.0

Terms of use:

- Licensed under CC-BY-NC-SA-4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/): non-commercial use only, and adapted material must be shared under the same licence.

Required credits:

- xBD: Gupta et al. (DIU / CMU SEI)
- Imagery: Maxar Open Data Program

Providers: DIU / CMU SEI, Maxar Open Data (Gupta et al.)

## Acknowledgements

TACO was designed by **César Aybar** and is specified at [https://asterisk.coop/taco/spec/](https://asterisk.coop/taco/spec/).

Built by [Oscar Pellicer](https://scholar.google.com/citations?hl=es&user=CCFJshwAAAAJ&view_op=list_works&sortby=pubdate) within the [ELLIOT](https://elliot-ai.eu) project at the
[Image and Signal Processing Group (ISP)](https://isp.uv.es/), Universitat de València.
