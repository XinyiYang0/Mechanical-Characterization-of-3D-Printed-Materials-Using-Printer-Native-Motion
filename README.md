# Mechanical Characterization of 3D-Printed Materials Using Printer-Native Motion

Research code accompanying the manuscript of the same title. The workflow uses
the printer head to excite, bend, unload, and ultimately fracture a printed
cantilever coupon on the build plate. Camera measurements and beam mechanics
are combined to reconstruct its effective mechanical response without measuring
the printer-head contact force.

**The methodological contribution is physics-based mechanical reconstruction
using printer-native motion.** Beam detection provides the measured geometry and
motion needed by that reconstruction; developing a new computer-vision detector
is not the central contribution.

## Experimental and computational workflow

1. **Print and cool.** Print a cantilever coupon with known dimensions and allow
   it to cool before testing. Measure its free length, cross section, and the
   condition-specific apparent density. The free length excludes the reinforced
   root transition.
2. **Elastic test.** Use printer-head motion to deflect and release the beam.
   Track tip displacement during free vibration and infer the effective modulus
   from its fundamental frequency and the known beam geometry. Fit the decay to
   estimate the effective damping ratio.
3. **Plastic test.** Bend beyond yield, record the loaded shape, unload, and
   record the residual shape. Reconstruct total and residual curvature with
   geometric and mechanical constraints. With the elastic modulus held fixed,
   identify constitutive parameters by minimizing the residual bending-moment
   imbalance after unloading.
4. **Fracture test.** Identify the final intact frame immediately before
   separation. Reconstruct its centerline and curvature, and calculate the
   outer-fiber strain as `epsilon_f = max(kappa_f) * thickness / 2`. Evaluate the
   calibrated constitutive relation at this strain to obtain fracture strength
   when the strain lies within the supported model range.
5. **Combine and validate.** Combine elastic, plastic, and fracture information
   into the effective coupon response. The manuscript compares the results
   with independent three-point-bending measurements. The reconstructed
   properties characterize the printed coupon and its fabrication condition;
   they should not be interpreted as intrinsic bulk-polymer constants.

The study's full elastoplastic reconstruction is demonstrated for PLA and PETG.
The linear-unloading assumption requires additional assessment for strongly
viscoelastic materials. Damping is a structural response metric.

## Repository contents

| Folder | Main entry points | Purpose |
| --- | --- | --- |
| [elastic](elastic/README.md) | `elastic_tracking.ipynb`, `response_fit.m` | Video tracking, damped response fitting, effective modulus |
| [plastic](plastic/README.md) | `curvature_extractor.m`, `general_model_fitter.m` | Physics-informed curvature and constitutive reconstruction |
| [fracture](fracture/README.md) | `fracture_curvature.ipynb`, `fracture_from_curvature.m` | Polynomial/spline fracture curvature; strain and strength from verified curvature |
| [tests](tests/README.md) | Python and MATLAB smoke checks | Synthetic checks of units, numerical behavior, and data interfaces |

The fracture image pipeline is the polynomial/spline algorithm used for the
reported fracture results, as confirmed by the author. The physics-informed
algorithm in the plastic step provides an option for future refinement of
fracture curvature; its use at the final intact state requires suitable inputs,
boundary conditions, and validation. Higher accuracy is not automatic.

## Installation and first run

Use Python 3.11 and MATLAB. Install the Python environment from this folder:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux instead: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open the elastic or fracture notebook, edit the clearly marked configuration
cell, and run the cells in order. Each module README specifies the input files,
units, parameters, outputs, and MATLAB commands. No input videos, images, or
experimental reference tables were included in the supplied `final_code` folder;
put your data in the documented locations before running an experiment.

MATLAB requirements:

- Elastic response fitting: **Curve Fitting Toolbox**.
- Plastic curvature extraction: **Image Processing Toolbox** and
  **Optimization Toolbox**.
- Parallel execution of the plastic scripts: **Parallel Computing Toolbox**
  (optional; serial execution is the default).
- Fracture property evaluation: base MATLAB; add `plastic/` to the MATLAB path
  when evaluating a fitted constitutive model.

The supplied plastic README reported testing on MATLAB R2025b and newer. The
release checks in [VALIDATION.md](docs/VALIDATION.md) record what was actually
checked locally, separately from full experimental reproduction.

## Image and video quality

**Image quality directly affects beam detection and therefore the reconstructed
mechanical quantities.** Prefer a clean background with clear contrast between
the beam and its surroundings, stable lighting, sharp focus, limited motion
blur, and a view that resolves the full tracked beam. Similar colors in the
background, occlusion, reflections, and compression artifacts can shift the
detected tip or centerline. Curvature is especially sensitive because it depends
on second spatial derivatives.

When the experimental environment does not permit a clean background, useful
options include a suitable region of interest, improved illumination or contrast,
background subtraction, removal of distracting objects in preprocessing, and
more capable segmentation or tracking algorithms. These are possible adaptations;
the supplied color-threshold detectors do not implement all of them. Check the
segmentation overlays and ensure preprocessing does not distort beam geometry.
The detector can be replaced while retaining the mechanical reconstruction,
provided the replacement produces calibrated, validated measurements.

For elastic tests, **use the physical camera acquisition rate**, not the exported
slow-motion playback rate. The manuscript uses 1000 fps acquisition even when
the exported video reports 25 fps. Misinterpreting this rate changes frequency
and, through its square, the inferred modulus. The required acquisition rate
depends on the beam's vibration frequency.

## Reproducibility and scope

- Notebooks are saved without execution outputs and contain one explicit workflow
  each. Experiment settings are separated from reusable processing functions.
- Missing tip detections retain their frame indices and timestamps. The code
  does not silently replace a failed detection with the previous measurement.
- The source elastic MATLAB model uses a fixed-phase damped cosine. The
  manuscript's separate FFT peak-refinement routine was not present in the
  supplied folder and is not represented as included here.
- The optional fracture curvature adjustment is identified as a heuristic and
  exported separately from raw curvature. See the fracture README before using it.
- Synthetic checks establish software behavior, not reproduction of the
  manuscript's experimental figures. Raw data and specimen-specific settings
  are needed for that comparison.

See [CHANGES.md](docs/CHANGES.md) for the source-to-release mapping and numerical
changes made during cleanup.

## Citation and credits

Please cite the accompanying manuscript, **Mechanical Characterization of
3D-Printed Materials Using Printer-Native Motion**, by Xinyi Yang, Xiaochen Li,
Chuqi Sun, Victor Riera Naranjo, Cole V Zachry, Dhanyaa Rajan, Christos E
Athanasiou, and Bolei Deng. Publication details should be taken from the final
published article when available; no article DOI is assigned here.

When using the plastic framework, also cite its separately archived software:

> Riera Naranjo, Victor. (2026). *Physics-Informed Curvature Extraction and
> Constitutive Modeling Framework* (v1) [Software]. Zenodo.
> https://doi.org/10.5281/zenodo.22163290

The [Zenodo record](https://zenodo.org/records/22163290) identifies Victor Riera
Naranjo as creator, Xinyi Yang as researcher, and Bolei Deng as supervisor.
The DOI identifies that plastic framework, not this entire repository.
Its original copyright and BSD 3-Clause license are retained. The included
CMA-ES implementation retains Nikolaus Hansen's notices and Victor Riera
Naranjo's modification credit. See [CITATIONS.bib](CITATIONS.bib),
[CITATION.cff](CITATION.cff), and [LICENSES.md](LICENSES.md).
