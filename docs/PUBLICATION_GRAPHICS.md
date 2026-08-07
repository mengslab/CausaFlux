# Publication graphics system

CausaFlux v1.4.0 uses one centralized visual and export contract for cancer, spatial, therapeutic, biomarker, active-learning, and neurobiology figures.

## Export profiles

| Profile | Width | Default height | Final font | Raster resolution |
|---|---:|---:|---:|---:|
| `nature_single` | 89 mm | 68 mm | 7 pt | 600 dpi |
| `nature_double` | 183 mm | 120 mm | 7.5 pt | 600 dpi |
| `cell_single` | 85 mm | 72 mm | 7 pt | 600 dpi |
| `cell_double` | 178 mm | 125 mm | 7.5 pt | 600 dpi |

All profiles use editable TrueType text in PDF, live text in SVG, white backgrounds, restrained colorblind-safe palettes, thin axes, and vector-first export.

## Per-panel artifact contract

For a figure named `therapeutic_ranking`, CausaFlux writes:

```text
therapeutic_ranking.png       600-dpi report raster
therapeutic_ranking.tiff      600-dpi LZW publication raster
therapeutic_ranking.svg       editable vector
therapeutic_ranking.pdf       editable vector
source_data/therapeutic_ranking__panel_a.csv
figure_manifests/therapeutic_ranking.json
```

The manifest records dimensions, profile, resolution, hashes, synthetic-data status, and panel source files.

## Graph grammar

- causal graphs use deterministic topological layers;
- node color represents biological object class;
- edge color represents sign only when sign is defined;
- communication overviews use saved circular coordinates for cell classes, not stochastic layouts;
- physical-proximity and molecular-communication graphs remain separate;
- edge width is mapped only to aggregate effect or communication strength;
- uncertainty is shown through intervals or opacity, never implied by attention weights;
- full node, edge, and layout tables accompany graph panels.

## Visual regression

`publication_graphics/visual_regression_baselines.csv` stores a perceptual hash, exact SHA-256, dimensions, profile, and tolerance for every panel.

```bash
causaflux publication-validate --input causaflux_v1.4.0_output
```

Perceptual hashes detect unexpected rendering changes while allowing narrowly configured antialiasing differences. Exact SHA-256 values remain available for provenance.

## Rebuilding figures

```bash
causaflux publication-build --input causaflux_v1.4.0_output
```

High-resolution domains are rendered in isolated subprocesses to avoid native-library and raster-buffer accumulation during one long scientific Python process.

## Interpretation

The bundled synthetic datasets and derived rankings are retained solely to verify figure generation, graph rendering, export integrity, and regression testing. They are not biological or clinical evidence.
