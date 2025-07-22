# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a well data repository containing oil and gas industry files for Murphy 1 well operations. The repository primarily stores:

- **LAS files** (.las): Log ASCII Standard format containing well log data
- **Mud logs** (PDF): Drilling progress and geological information
- **Well reports** (PDF): Comprehensive drilling and completion reports

## File Types and Formats

### LAS Files
- Industry standard format for well log data
- Contains header information and tabular log data
- Can be parsed using petroleum engineering libraries if needed

### PDF Documents
- Well reports and mud logs in PDF format
- May contain charts, graphs, and tabular data
- Often contain sensitive operational information

## Well Log Interpretation and Processing

This repository focuses on petrophysical analysis and well log interpretation. When processing well data:

### Lithology Identification
- **Gamma Ray (GR)**: High values indicate shale (clay-rich), low values suggest clean sandstone/carbonate
- **Spontaneous Potential (SP)**: Negative deflections indicate permeable sands, baseline suggests shales
- **Cross-plot Analysis**: Use neutron-density overlays for lithology determination:
  - Sandstone: Neutron and density porosities align closely
  - Shale: High neutron, lower density porosity
  - Limestone/Dolomite: Density porosity higher than neutron in dolomites
  - Gas effect: Low neutron, high density (light hydrocarbons)

### Key Calculations and Formulas
- **Shale Volume**: V_shale = (GR - GR_min) / (GR_max - GR_min) (Larionov equation)
- **Porosity from Density**: phi_D = (rho_matrix - rho_bulk) / (rho_matrix - rho_fluid)
- **Water Saturation**: S_w = (a * R_w / (phi^m * R_t))^(1/n) (Archie's equation)

### Rock Characteristics Analysis
- **Porosity**: Calculate from density and neutron logs, consider total vs. effective porosity
- **Permeability**: Infer from SP deflections combined with resistivity patterns
- **Fluid Saturation**: Integrate resistivity and porosity data
- **Formation Boundaries**: Identify using gamma ray and SP log responses

### Data Processing Approach
1. Parse LAS files for header information and log curve data
2. Apply quality control checks for data integrity
3. Calculate derived properties (shale volume, porosity, saturation)
4. Generate cross-plots for lithology identification
5. Create interpretation flags and zone classifications
6. Consider calibration with core data when available

### Working with This Repository

Since this is a data repository rather than a software project, typical development commands (build, test, lint) are not applicable. When working with these files:

1. **Petrophysical Analysis**: Focus on standard geophysical interpretations using established principles
2. **LAS File Processing**: Parse header and curve data, maintain format integrity
3. **Visualization**: Create plots for log curves, cross-plots, and derived properties
4. **Confidentiality**: Well data may contain proprietary or sensitive information

## Notes

This repository does not contain traditional software development files (package.json, source code, etc.). All operations should focus on data management and analysis rather than code development.