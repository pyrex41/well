# Well Log Analysis Tool

Advanced LAS file analysis and petrophysical interpretation tool for Murphy #1 Well in South Johnson Field.

## Features

- **Comprehensive LAS file parsing** with robust error handling
- **Multi-method lithology identification** (Gamma Ray, Neutron-Density, PE, ML clustering)
- **Advanced petrophysical calculations** (porosity, saturation, permeability)
- **State-of-the-art visualizations** (well logs, cross-plots, histograms)
- **Automated report generation** with reservoir quality assessment

## Installation

### 1. Install UV (if not already installed)

UV is a fast Python package manager. Choose one of the following methods:

**📖 Full installation guide**: [UV Installation Documentation](https://docs.astral.sh/uv/getting-started/installation/)

#### macOS/Linux (using curl):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### macOS (using Homebrew):
```bash
brew install uv
```

#### Windows (using PowerShell):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Alternative (using pip):
```bash
pip install uv
```

### 2. Set up the project environment

```bash
# Navigate to the project directory
cd /path/to/well-analysis

# Sync dependencies (creates virtual environment automatically)
uv sync
```

## Usage

### Run the complete analysis:

```bash
uv run python las_analysis.py
```

This will:
1. Load and parse the Murphy #1 LAS file
2. Perform comprehensive data quality assessment
3. Execute multi-method lithology identification
4. Calculate petrophysical properties (porosity, saturation, permeability)
5. Generate visualizations and cross-plots
6. Create a detailed analysis report

### Generated Output Files:

- `well_logs_display.png` - Standard log curve display
- `crossplots_analysis.png` - Petrophysical cross-plot analysis
- `histograms_analysis.png` - Statistical distribution plots
- `interpretation_summary.png` - Integrated interpretation tracks
- `Murphy1_Analysis_Report.txt` - Comprehensive written report

## Analysis Methods

### Lithology Identification
- **Gamma Ray Analysis**: Larionov equations for shale volume
- **Neutron-Density Cross-plots**: Matrix identification
- **Photoelectric Factor**: Mineral classification
- **Machine Learning**: K-means clustering for facies

### Petrophysical Calculations  
- **Porosity**: Multi-log integration with gas correction
- **Water Saturation**: Archie's equation implementation
- **Permeability**: Kozeny-Carman and Timur correlations
- **Net-to-Gross**: Reservoir development assessment

## Key Results Summary

Based on the Murphy #1 analysis:

- **Primary Lithology**: 71% Clean Sandstone, 15% Shale, 14% Shaly Sandstone
- **Average Porosity**: 10.2% (good reservoir quality)
- **Water Saturation**: 79.4% (20.6% hydrocarbon saturation)
- **Average Permeability**: 45.7 mD (fair to good)
- **Net-to-Gross Ratio**: 71.7% (excellent reservoir development)

## Calibration Requirements

**Important**: The following parameters should be calibrated with local data:

- **Archie Parameters**: Formation water resistivity (Rw), cementation exponent (m), saturation exponent (n)
- **Matrix Properties**: Rock densities based on XRD analysis
- **Cutoff Values**: Net pay and lithology cutoffs from core description

## File Structure

```
well-analysis/
├── README.md                          # This file
├── CODE_DESCRIPTION.md                # Detailed code documentation
├── CLAUDE.md                          # Claude Code guidance
├── pyproject.toml                     # Project configuration
├── las_analysis.py                    # Main analysis script
├── Murphy 1_MainPass.las              # LAS file data
├── W007 Murphy 1 Mud Log.pdf          # Mud log report
├── W007 Murphy 1 Well Report 12-17.pdf # Well completion report
└── [generated output files]           # Analysis results
```

## Requirements

- Python ≥3.9
- Dependencies managed via UV (see pyproject.toml)

## Troubleshooting

If you encounter issues:

1. **UV not found**: Ensure UV is installed and in your PATH
2. **Permission errors**: Use appropriate file permissions for the project directory  
3. **Missing dependencies**: Run `uv sync` to ensure all packages are installed
4. **LAS file errors**: Verify the LAS file path and format

## Further Development

See `CODE_DESCRIPTION.md` for detailed technical documentation and recommendations for advanced analysis enhancements.