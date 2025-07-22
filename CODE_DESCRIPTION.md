# LAS Analysis Script - Detailed Code Description

## Overview

The `las_analysis.py` script is a comprehensive well log analysis tool that performs state-of-the-art petrophysical interpretation of LAS (Log ASCII Standard) files. It combines multiple industry-standard methods for lithology identification, rock property calculations, and reservoir characterization.

## Script Structure

### Imports and Dependencies

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import lasio (optional)
```

**Dependencies:**
- **numpy**: Numerical computations and array operations
- **pandas**: Data manipulation and analysis
- **matplotlib**: Static plotting and visualization
- **seaborn**: Statistical data visualization
- **scipy**: Scientific computing (statistics, optimization)
- **scikit-learn**: Machine learning algorithms for clustering
- **lasio**: LAS file parsing library (optional, falls back to manual parsing)

---

## Main Class: PetrophysicalAnalyzer

The core class that encapsulates all well log analysis functionality.

### Class Initialization

```python
def __init__(self, las_file_path):
```

**Purpose**: Initialize the analyzer with a LAS file path.

**Inputs:**
- `las_file_path` (str): Absolute path to the LAS file

**Outputs:**
- Initializes instance variables:
  - `self.las_file_path`: Path to LAS file
  - `self.las`: LAS file object (None initially)
  - `self.df`: DataFrame containing log data (None initially)
  - `self.well_info`: Dictionary of well header information
  - `self.curves`: Dictionary of curve objects

---

### Data Loading Methods

#### `load_las_file()`

**Purpose**: Load and parse LAS file with comprehensive error handling.

**Inputs:** None (uses `self.las_file_path`)

**Outputs:**
- **Returns**: Boolean (True if successful, False if failed)
- **Side Effects**: 
  - Populates `self.las`, `self.df`, `self.well_info`, `self.curves`
  - Prints loading status and basic file information

**Algorithm:**
1. Attempts to use `lasio` library for parsing
2. Falls back to manual parsing if `lasio` unavailable
3. Reports data range and available curves
4. Handles various file format exceptions

**Assumptions:**
- LAS file follows standard format
- Depth is the first column/index
- Missing values are represented as -999.250

#### `_manual_las_parse()`

**Purpose**: Fallback method for parsing LAS files without lasio library.

**Inputs:** None (reads from `self.las_file_path`)

**Outputs:**
- **Side Effects**: Populates `self.df` with parsed data
- Replaces null values (-999.25) with NaN

**Algorithm:**
1. Reads file line by line
2. Identifies ASCII data section marker (~ASCII or ~A)
3. Extracts curve names from header
4. Parses numerical data, handling missing values
5. Creates DataFrame with depth as index

**Assumptions:**
- Standard LAS file structure
- ASCII data section clearly marked
- Consistent number of columns throughout data section

---

### Data Quality Assessment

#### `data_quality_assessment()`

**Purpose**: Comprehensive evaluation of data quality and completeness.

**Inputs:** None (uses `self.df`)

**Outputs:**
- **Returns**: None
- **Side Effects**: Prints detailed quality assessment report

**Analysis Components:**
1. **Basic Statistics**:
   - Total data points
   - Depth range and sampling interval
   
2. **Data Completeness**:
   - Percentage of valid data points per curve
   - Identifies curves with significant missing data

3. **Outlier Detection**:
   - Uses Interquartile Range (IQR) method
   - Formula: Outliers = values < Q1 - 1.5×IQR or > Q3 + 1.5×IQR
   - Reports percentage of outliers per curve

**Parameter Assumptions:**
- IQR multiplier: 1.5 (standard statistical practice)
- Missing value threshold: -999.25

---

### Lithology Identification Methods

#### `lithology_identification()`

**Purpose**: Master function that applies multiple lithology identification methods.

**Inputs:** None (uses `self.df`)

**Outputs:**
- **Returns**: Dictionary containing results from all applied methods
- Keys may include: 'gamma_ray', 'neutron_density', 'photoelectric', 'ml_clustering'

**Methods Applied:**
1. Gamma Ray classification (if GR data available)
2. Neutron-Density cross-plot analysis
3. Photoelectric Factor analysis
4. Machine Learning clustering

#### `_gamma_ray_lithology(gr_col)`

**Purpose**: Classify lithology using gamma ray log and calculate shale volume.

**Inputs:**
- `gr_col` (str): Column name for gamma ray data

**Outputs:**
- **Returns**: Dictionary with keys:
  - 'lithology': Pandas Series with lithology classifications
  - 'v_shale_tertiary': Shale volume for Tertiary rocks
  - 'v_shale_older': Shale volume for older rocks
  - 'gr_clean': Clean sand baseline
  - 'gr_shale': Pure shale value

**Algorithm:**
1. **Baseline Determination**:
   - Clean sand GR = 5th percentile of data
   - Pure shale GR = 95th percentile of data

2. **Normalization**:
   - GR_norm = (GR - GR_clean) / (GR_shale - GR_clean)

3. **Shale Volume Calculation** (Larionov equations):
   - **Tertiary**: V_shale = 0.083 × (2^(3.7 × GR_norm) - 1)
   - **Older rocks**: V_shale = 0.33 × (2^(2 × GR_norm) - 1)

4. **Classification**:
   - Clean Sandstone: V_shale < 0.15
   - Shaly Sandstone: 0.15 ≤ V_shale < 0.50
   - Shale: V_shale ≥ 0.50

**Parameter Assumptions:**
- Larionov equation coefficients (industry standard)
- Classification cutoffs: 0.15 and 0.50 for shale volume

#### `_neutron_density_lithology(nphi_col, rhob_col)`

**Purpose**: Determine lithology using neutron-density cross-plot analysis.

**Inputs:**
- `nphi_col` (str): Neutron porosity column name
- `rhob_col` (str): Bulk density column name

**Outputs:**
- **Returns**: Dictionary with keys:
  - 'lithology': Pandas Series with lithology assignments
  - 'nphi_data': Processed neutron data
  - 'rhob_data': Processed density data

**Algorithm:**
1. **Matrix Point Definitions**:
   - Sandstone: φ = 0.0, ρ = 2.65 g/cc
   - Limestone: φ = 0.0, ρ = 2.71 g/cc
   - Dolomite: φ = 0.0, ρ = 2.87 g/cc
   - Anhydrite: φ = 0.0, ρ = 2.96 g/cc

2. **Distance Calculation**:
   - Euclidean distance to each matrix point
   - Density weighted by factor 0.1 to balance scales

3. **Gas Detection**:
   - Gas Sand: ρ < 2.3 g/cc AND φ < 0.15

**Parameter Assumptions:**
- Standard matrix densities from literature
- Gas detection thresholds: density < 2.3 g/cc, neutron < 15%
- Distance weighting factor: 0.1 for density

#### `_photoelectric_lithology(pe_col)`

**Purpose**: Identify minerals using Photoelectric Factor (PE) values.

**Inputs:**
- `pe_col` (str): PE column name

**Outputs:**
- **Returns**: Dictionary with keys:
  - 'lithology': Pandas Series with mineral identifications
  - 'pe_data': PE values

**PE Value Ranges** (barns/electron):
- Quartz/Sandstone: 1.8 - 1.9
- Calcite/Limestone: 5.0 - 5.2
- Dolomite: 3.0 - 3.2
- Clay/Shale: 2.8 - 3.3
- Anhydrite: 5.0 - 5.1
- Salt: 4.6 - 4.8

**Parameter Assumptions:**
- PE ranges from published literature
- Values outside ranges classified as "Unknown"

#### `_ml_lithology_clustering()`

**Purpose**: Apply machine learning clustering for facies classification.

**Inputs:** None (uses available log curves)

**Outputs:**
- **Returns**: Dictionary with keys:
  - 'lithology': Pandas Series with cluster assignments
  - 'clusters': Cluster labels
  - 'scaler': StandardScaler object
  - 'features': List of curves used

**Algorithm:**
1. **Feature Selection**: Uses curves ['GGCE', 'NPRL', 'DEN', 'PDPE', 'RTAT']
2. **Data Preprocessing**: StandardScaler normalization
3. **Optimal Cluster Determination**: Elbow method on K=2 to 7
4. **K-Means Clustering**: Final clustering with optimal K

**Parameter Assumptions:**
- K-means random state: 42 (reproducibility)
- Minimum clusters: 2, maximum: 7
- Requires minimum 10 data points

---

### Petrophysical Calculations

#### `petrophysical_calculations()`

**Purpose**: Master function for calculating rock properties.

**Inputs:** None (uses `self.df`)

**Outputs:**
- **Returns**: Dictionary with calculated properties
- Keys may include: 'porosity', 'water_saturation', 'permeability', 'net_to_gross'

#### `_calculate_porosity()`

**Purpose**: Calculate porosity using neutron and density logs.

**Inputs:** None (uses 'NPRL' and 'DEN' columns)

**Outputs:**
- **Returns**: Dictionary with keys:
  - 'phi_density': Density-derived porosity
  - 'phi_neutron': Neutron porosity
  - 'phi_combined': Average porosity
  - 'phi_corrected': Gas-corrected porosity
  - 'gas_flag': Boolean array indicating gas zones

**Formulas:**
1. **Density Porosity**: φ_D = (ρ_matrix - ρ_bulk) / (ρ_matrix - ρ_fluid)
2. **Combined Porosity**: φ_avg = (φ_neutron + φ_density) / 2
3. **Gas Correction**: Use density porosity when φ_neutron < φ_density - 4%

**Parameter Assumptions:**
- Matrix density (ρ_matrix): 2.65 g/cc (sandstone)
- Fluid density (ρ_fluid): 1.0 g/cc (water)
- Porosity limits: 0 to 50%
- Gas effect threshold: 4% porosity unit difference

#### `_calculate_water_saturation(porosity_data)`

**Purpose**: Calculate water saturation using Archie's equation.

**Inputs:**
- `porosity_data` (dict): Output from `_calculate_porosity()`

**Outputs:**
- **Returns**: Dictionary with keys:
  - 'sw': Water saturation
  - 'sh': Hydrocarbon saturation (1 - Sw)
  - 'formation_factor': Formation factor
  - 'archie_params': Archie equation parameters

**Archie's Equation**: Sw = ((a × Rw) / (φ^m × Rt))^(1/n)

**Parameter Assumptions:**
- Formation water resistivity (Rw): 0.1 ohm-m
- Tortuosity factor (a): 1.0
- Cementation exponent (m): 2.0
- Saturation exponent (n): 2.0

**Note**: These parameters should be calibrated with local core data.

#### `_estimate_permeability(porosity_data)`

**Purpose**: Estimate permeability using empirical correlations.

**Inputs:**
- `porosity_data` (dict): Output from `_calculate_porosity()`

**Outputs:**
- **Returns**: Dictionary with keys:
  - 'k_kozeny': Kozeny-Carman permeability (mD)
  - 'k_timur': Timur correlation permeability (mD)
  - 'k_average': Average of both methods (mD)

**Formulas:**
1. **Kozeny-Carman**: K = 5000 × φ³ / (1-φ)²
2. **Timur**: K = 0.136 × φ^4.4 / (1-φ)²

**Parameter Assumptions:**
- Coefficients from published correlations
- Results in millidarcies (mD)

#### `_calculate_net_to_gross()`

**Purpose**: Calculate net reservoir thickness ratio.

**Inputs:** None (uses 'GGCE' column)

**Outputs:**
- **Returns**: Dictionary with keys:
  - 'net_flag': Boolean array for net reservoir
  - 'ntg_ratio': Net-to-gross ratio
  - 'net_thickness': Net reservoir thickness (ft)
  - 'gross_thickness': Total logged thickness (ft)

**Formula**: N/G = Net_thickness / Gross_thickness

**Parameter Assumptions:**
- GR cutoff: 75 GAPI (net reservoir threshold)
- Sampling interval: 0.5 ft

---

### Visualization Methods

#### `create_visualizations(results=None)`

**Purpose**: Generate comprehensive visualization suite.

**Inputs:**
- `results` (dict, optional): Results from petrophysical calculations

**Outputs:**
- **Side Effects**: Creates and saves multiple PNG files
- Calls individual plotting methods

#### `_plot_well_logs()`

**Purpose**: Create standard well log display with multiple tracks.

**Inputs:** None (uses `self.df`)

**Outputs:**
- **Side Effects**: Saves 'well_logs_display.png'

**Features:**
- Multiple tracks for different log types
- Logarithmic scaling for resistivity
- Proper depth scaling (inverted y-axis)
- Color coding per industry standards

**Track Configuration:**
- GGCE: Green, linear scale
- SPCG: Blue, linear scale  
- RTAT: Red, log scale
- NPRL: Blue, linear scale
- DEN: Red, linear scale

#### `_plot_crossplots()`

**Purpose**: Create petrophysical cross-plot analysis.

**Inputs:** None (uses `self.df`)

**Outputs:**
- **Side Effects**: Saves 'crossplots_analysis.png'

**Cross-plots Generated:**
1. **Neutron-Density**: Lithology identification
2. **PE-Density**: Mineral identification
3. **Resistivity-Porosity**: Saturation analysis
4. **GR-SP**: Formation evaluation

**Features:**
- Color-coded by depth
- Logarithmic scaling where appropriate
- Industry-standard axis orientations

#### `_plot_histograms()`

**Purpose**: Statistical distribution analysis of log data.

**Inputs:** None (uses `self.df`)

**Outputs:**
- **Side Effects**: Saves 'histograms_analysis.png'

**Features:**
- 50-bin histograms for key parameters
- Statistical overlays (mean, standard deviation)
- Frequency distributions with outlier indication

#### `_plot_interpretation_track(results)`

**Purpose**: Create integrated interpretation display.

**Inputs:**
- `results` (dict): Combined results from all analyses

**Outputs:**
- **Side Effects**: Saves 'interpretation_summary.png'

**Tracks:**
1. **Lithology**: Color-coded rock types
2. **Porosity**: Blue-filled porosity curve
3. **Saturation**: Water (blue) and hydrocarbon (red) saturation
4. **Permeability**: Log-scale permeability display

---

### Reporting Method

#### `generate_report(results)`

**Purpose**: Generate comprehensive text report of analysis.

**Inputs:**
- `results` (dict): Combined results from all analyses

**Outputs:**
- **Returns**: Report text string
- **Side Effects**: Saves 'Murphy1_Analysis_Report.txt'

**Report Sections:**
1. **Well Information**: Header data from LAS file
2. **Data Summary**: Basic statistics and data quality
3. **Lithology Analysis**: Rock type distributions
4. **Petrophysical Properties**: Average porosity, saturation, permeability
5. **Reservoir Quality Assessment**: Quality classifications

**Quality Classifications:**
- Excellent: φ > 15%
- Good: φ = 10-15%
- Fair: φ = 5-10%  
- Poor: φ < 5%

---

## Main Function

### `main()`

**Purpose**: Orchestrate the complete analysis workflow.

**Inputs:** None

**Outputs:**
- **Side Effects**: Complete analysis with all outputs

**Workflow:**
1. Initialize PetrophysicalAnalyzer
2. Load LAS file
3. Assess data quality
4. Perform lithology identification
5. Calculate petrophysical properties
6. Create visualizations
7. Generate report
8. Display summary and recommendations

---

## Key Parameter Assumptions and Calibration Requirements

### Critical Assumptions Requiring Local Calibration:

1. **Archie Equation Parameters**:
   - Formation water resistivity (Rw): 0.1 ohm-m
   - Cementation exponent (m): 2.0
   - Saturation exponent (n): 2.0
   - **Calibration**: Core analysis, formation water samples

2. **Matrix Properties**:
   - Sandstone density: 2.65 g/cc
   - Limestone density: 2.71 g/cc
   - **Calibration**: X-ray diffraction, core analysis

3. **Cutoff Values**:
   - Net pay GR cutoff: 75 GAPI
   - Shale volume cutoffs: 15%, 50%
   - **Calibration**: Core description, production data

4. **Permeability Correlations**:
   - Kozeny-Carman coefficient: 5000
   - Timur coefficients: 0.136, 4.4
   - **Calibration**: Core permeability measurements

### Industry Standard Values Used:

1. **PE Values**: From published literature (Schlumberger, Halliburton charts)
2. **Larionov Coefficients**: Standard published values
3. **Statistical Methods**: IQR outlier detection with 1.5 multiplier
4. **Data Quality**: -999.25 as null value indicator

---

## Error Handling and Robustness

### Implemented Safeguards:

1. **Missing Data**: Graceful handling of NaN values
2. **File Format**: Fallback manual parsing if lasio unavailable  
3. **Physical Constraints**: Clipping porosity and saturation to realistic ranges
4. **Minimum Data Requirements**: Checks for sufficient data points
5. **Curve Availability**: Dynamic method selection based on available curves

### Limitations:

1. **Complex Lithologies**: Mixed mineralogy may require advanced models
2. **Invasion Effects**: No correction for mud filtrate invasion
3. **Temperature Correction**: Limited temperature compensation
4. **Anisotropy**: Assumes isotropic rock properties

---

## Recommended Enhancements:

1. **Core Calibration Module**: Integrate core data for parameter calibration
2. **Advanced Saturation Models**: Waxman-Smits for shaly sands
3. **NMR Integration**: Add nuclear magnetic resonance data if available
4. **Fracture Detection**: Advanced resistivity image interpretation
5. **Uncertainty Quantification**: Monte Carlo error propagation
6. **Real-time Updates**: Parameter adjustment based on production data