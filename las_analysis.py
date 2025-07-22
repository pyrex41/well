#!/usr/bin/env python3
"""
Advanced LAS File Analysis and Petrophysical Interpretation
State-of-the-art well log analysis with comprehensive lithology identification,
rock property calculations, and advanced interpretation algorithms.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import lasio
except ImportError:
    print("Warning: lasio not available. Install with: pip install lasio")
    lasio = None

class PetrophysicalAnalyzer:
    """Advanced petrophysical analysis of well log data."""
    
    def __init__(self, las_file_path):
        """Initialize with LAS file path."""
        self.las_file_path = las_file_path
        self.las = None
        self.df = None
        self.well_info = {}
        self.curves = {}
        
    def load_las_file(self):
        """Load and parse LAS file with comprehensive error handling."""
        try:
            if lasio is None:
                # Manual LAS parsing fallback
                self._manual_las_parse()
            else:
                self.las = lasio.read(self.las_file_path)
                self.df = self.las.df()
                self.well_info = dict(self.las.well)
                self.curves = {curve.mnemonic: curve for curve in self.las.curves}
                
            print(f"✅ Successfully loaded LAS file: {self.las_file_path}")
            print(f"📊 Data range: {self.df.index.min():.1f} to {self.df.index.max():.1f} ft")
            print(f"📈 Available curves: {list(self.df.columns)}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading LAS file: {e}")
            return False
    
    def _manual_las_parse(self):
        """Manual LAS parsing when lasio is not available."""
        with open(self.las_file_path, 'r') as f:
            lines = f.readlines()
        
        # Find ASCII data section
        data_start = None
        header_info = {}
        curves = []
        
        for i, line in enumerate(lines):
            if line.strip().startswith('~ASCII') or line.strip().startswith('~A'):
                data_start = i + 1
                break
            elif line.strip().startswith('DEPT'):
                # Parse curve names from header
                parts = line.strip().split()
                if len(parts) > 0:
                    curves = parts
        
        if data_start and curves:
            # Read data
            data_lines = [line.strip() for line in lines[data_start:] if line.strip() and not line.startswith('#')]
            data = []
            
            for line in data_lines:
                values = line.split()
                if len(values) == len(curves):
                    try:
                        row = [float(v) if v != '-999.250000' else np.nan for v in values]
                        data.append(row)
                    except ValueError:
                        continue
            
            if data:
                self.df = pd.DataFrame(data, columns=curves)
                self.df.set_index(curves[0], inplace=True)  # First column as depth index
                self.df.replace(-999.25, np.nan, inplace=True)
                print("📋 Manual LAS parsing completed")
    
    def data_quality_assessment(self):
        """Comprehensive data quality assessment."""
        print("\n🔍 DATA QUALITY ASSESSMENT")
        print("="*50)
        
        # Basic statistics
        total_points = len(self.df)
        depth_range = self.df.index.max() - self.df.index.min()
        
        print(f"Total data points: {total_points:,}")
        print(f"Depth range: {self.df.index.min():.1f} to {self.df.index.max():.1f} ft ({depth_range:.1f} ft)")
        print(f"Sample interval: {np.median(np.diff(self.df.index)):.2f} ft")
        
        # Null value analysis
        print("\n📊 DATA COMPLETENESS BY CURVE:")
        for col in self.df.columns:
            valid_count = self.df[col].count()
            completeness = (valid_count / total_points) * 100
            print(f"{col:<8}: {valid_count:>6,} points ({completeness:>5.1f}% complete)")
        
        # Identify outliers using IQR method
        print("\n⚠️  OUTLIER DETECTION:")
        for col in self.df.select_dtypes(include=[np.number]).columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | 
                       (self.df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                print(f"{col:<8}: {outliers:>6} outliers ({outliers/self.df[col].count()*100:.1f}%)")
    
    def lithology_identification(self):
        """Advanced lithology identification using multiple methods."""
        print("\n🗿 LITHOLOGY IDENTIFICATION")
        print("="*50)
        
        results = {}
        
        # Method 1: Gamma Ray based classification
        if 'GGCE' in self.df.columns:
            gr_col = 'GGCE'
        elif 'GR' in self.df.columns:
            gr_col = 'GR'
        else:
            gr_col = None
            
        if gr_col:
            results['gamma_ray'] = self._gamma_ray_lithology(gr_col)
        
        # Method 2: Neutron-Density cross-plot analysis
        if 'NPRL' in self.df.columns and 'DEN' in self.df.columns:
            results['neutron_density'] = self._neutron_density_lithology('NPRL', 'DEN')
        
        # Method 3: PE (Photoelectric Factor) analysis
        if 'PDPE' in self.df.columns:
            results['photoelectric'] = self._photoelectric_lithology('PDPE')
        
        # Method 4: Machine Learning clustering
        if len(self.df.select_dtypes(include=[np.number]).columns) >= 3:
            results['ml_clustering'] = self._ml_lithology_clustering()
        
        return results
    
    def _gamma_ray_lithology(self, gr_col):
        """Lithology from Gamma Ray using Larionov method."""
        gr_data = self.df[gr_col].dropna()
        
        # Define cutoffs (can be calibrated based on local geology)
        gr_clean = gr_data.quantile(0.05)  # Clean sand baseline
        gr_shale = gr_data.quantile(0.95)  # Pure shale
        
        # Larionov equation for shale volume
        gr_normalized = (gr_data - gr_clean) / (gr_shale - gr_clean)
        gr_normalized = gr_normalized.clip(0, 1)
        
        # Tertiary rocks (Larionov, 1969)
        v_shale_tertiary = 0.083 * (2**(3.7 * gr_normalized) - 1)
        
        # Older rocks (Larionov, 1969)
        v_shale_older = 0.33 * (2**(2 * gr_normalized) - 1)
        
        # Classification
        lithology = pd.Series(index=gr_data.index, dtype='object')
        lithology[v_shale_tertiary < 0.15] = 'Clean Sandstone'
        lithology[(v_shale_tertiary >= 0.15) & (v_shale_tertiary < 0.50)] = 'Shaly Sandstone'
        lithology[v_shale_tertiary >= 0.50] = 'Shale'
        
        print(f"Gamma Ray Analysis ({gr_col}):")
        print(f"  Clean sand GR: {gr_clean:.1f} GAPI")
        print(f"  Pure shale GR: {gr_shale:.1f} GAPI")
        print(f"  Lithology distribution:")
        for lith, count in lithology.value_counts().items():
            percentage = (count / len(lithology)) * 100
            print(f"    {lith}: {percentage:.1f}%")
        
        return {
            'lithology': lithology,
            'v_shale_tertiary': v_shale_tertiary,
            'v_shale_older': v_shale_older,
            'gr_clean': gr_clean,
            'gr_shale': gr_shale
        }
    
    def _neutron_density_lithology(self, nphi_col, rhob_col):
        """Neutron-Density cross-plot lithology identification."""
        valid_data = self.df[[nphi_col, rhob_col]].dropna()
        
        if len(valid_data) == 0:
            return None
            
        nphi = valid_data[nphi_col] / 100  # Convert to fraction if in percentage
        rhob = valid_data[rhob_col]
        
        # Matrix points (typical values)
        matrices = {
            'Sandstone': {'nphi': 0.0, 'rhob': 2.65},
            'Limestone': {'nphi': 0.0, 'rhob': 2.71},
            'Dolomite': {'nphi': 0.0, 'rhob': 2.87},
            'Anhydrite': {'nphi': 0.0, 'rhob': 2.96}
        }
        
        # Calculate distances to matrix points
        lithology = pd.Series(index=valid_data.index, dtype='object')
        
        for idx in valid_data.index:
            distances = {}
            for matrix, props in matrices.items():
                dist = np.sqrt((nphi[idx] - props['nphi'])**2 + 
                              (rhob[idx] - props['rhob'])**2 * 0.1)  # Weight density less
                distances[matrix] = dist
            
            # Assign closest matrix
            lithology[idx] = min(distances.keys(), key=lambda k: distances[k])
            
            # Gas effect detection (low density, low neutron)
            if rhob[idx] < 2.3 and nphi[idx] < 0.15:
                lithology[idx] = 'Gas Sand'
        
        print(f"\nNeutron-Density Analysis:")
        for lith, count in lithology.value_counts().items():
            percentage = (count / len(lithology)) * 100
            print(f"  {lith}: {percentage:.1f}%")
        
        return {
            'lithology': lithology,
            'nphi_data': nphi,
            'rhob_data': rhob
        }
    
    def _photoelectric_lithology(self, pe_col):
        """Lithology identification using Photoelectric Factor."""
        pe_data = self.df[pe_col].dropna()
        
        # PE values for common minerals
        pe_ranges = {
            'Quartz/Sandstone': (1.8, 1.9),
            'Calcite/Limestone': (5.0, 5.2),
            'Dolomite': (3.0, 3.2),
            'Clay/Shale': (2.8, 3.3),
            'Anhydrite': (5.0, 5.1),
            'Salt': (4.6, 4.8)
        }
        
        lithology = pd.Series(index=pe_data.index, dtype='object')
        
        for mineral, (pe_min, pe_max) in pe_ranges.items():
            mask = (pe_data >= pe_min) & (pe_data <= pe_max)
            lithology[mask] = mineral
        
        # Unidentified values
        lithology[lithology.isnull()] = 'Unknown'
        
        print(f"\nPhotoelectric Factor Analysis:")
        for lith, count in lithology.value_counts().items():
            percentage = (count / len(lithology)) * 100
            print(f"  {lith}: {percentage:.1f}%")
        
        return {'lithology': lithology, 'pe_data': pe_data}
    
    def _ml_lithology_clustering(self):
        """Machine learning based lithology clustering."""
        # Select relevant curves for clustering
        analysis_curves = []
        for curve in ['GGCE', 'NPRL', 'DEN', 'PDPE', 'RTAT']:
            if curve in self.df.columns:
                analysis_curves.append(curve)
        
        if len(analysis_curves) < 2:
            return None
        
        # Prepare data
        ml_data = self.df[analysis_curves].dropna()
        if len(ml_data) < 10:
            return None
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(ml_data)
        
        # Determine optimal number of clusters using elbow method
        inertias = []
        k_range = range(2, min(8, len(ml_data)//5))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)
        
        # Simple elbow detection
        optimal_k = k_range[np.argmax(np.diff(np.diff(inertias)))] if len(inertias) > 2 else 3
        
        # Final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        lithology = pd.Series(index=ml_data.index, dtype='object')
        for i, cluster in enumerate(clusters):
            lithology.iloc[i] = f'Facies_{cluster}'
        
        print(f"\nMachine Learning Clustering (K={optimal_k}):")
        for lith, count in lithology.value_counts().items():
            percentage = (count / len(lithology)) * 100
            print(f"  {lith}: {percentage:.1f}%")
        
        return {
            'lithology': lithology,
            'clusters': clusters,
            'scaler': scaler,
            'features': analysis_curves
        }
    
    def petrophysical_calculations(self):
        """Advanced petrophysical property calculations."""
        print("\n📊 PETROPHYSICAL CALCULATIONS")
        print("="*50)
        
        results = {}
        
        # Porosity calculations
        if 'NPRL' in self.df.columns and 'DEN' in self.df.columns:
            results['porosity'] = self._calculate_porosity()
        
        # Water saturation (Archie's equation)
        if 'RTAT' in self.df.columns and 'porosity' in results:
            results['water_saturation'] = self._calculate_water_saturation(results['porosity'])
        
        # Permeability estimation
        if 'porosity' in results:
            results['permeability'] = self._estimate_permeability(results['porosity'])
        
        # Net-to-gross calculation
        if 'GGCE' in self.df.columns:
            results['net_to_gross'] = self._calculate_net_to_gross()
        
        return results
    
    def _calculate_porosity(self):
        """Calculate porosity from neutron and density logs."""
        nphi = self.df['NPRL'] / 100 if 'NPRL' in self.df.columns else None
        rhob = self.df['DEN'] if 'DEN' in self.df.columns else None
        
        if nphi is None or rhob is None:
            return None
        
        # Matrix densities (can be depth-dependent based on lithology)
        rho_matrix = 2.65  # Sandstone default
        rho_fluid = 1.0    # Water
        
        # Density porosity
        phi_density = (rho_matrix - rhob) / (rho_matrix - rho_fluid)
        phi_density = phi_density.clip(0, 0.5)  # Reasonable limits
        
        # Neutron porosity (already in fraction)
        phi_neutron = nphi.clip(0, 0.5)
        
        # Combined porosity (average, with gas correction)
        phi_combined = (phi_density + phi_neutron) / 2
        
        # Gas effect correction (when neutron < density)
        gas_flag = phi_neutron < (phi_density - 0.04)
        phi_corrected = phi_combined.copy()
        phi_corrected[gas_flag] = phi_density[gas_flag]  # Use density porosity for gas zones
        
        print("Porosity Analysis:")
        print(f"  Average density porosity: {phi_density.mean():.3f} ({phi_density.mean()*100:.1f}%)")
        print(f"  Average neutron porosity: {phi_neutron.mean():.3f} ({phi_neutron.mean()*100:.1f}%)")
        print(f"  Average combined porosity: {phi_corrected.mean():.3f} ({phi_corrected.mean()*100:.1f}%)")
        print(f"  Possible gas zones: {gas_flag.sum()} points ({gas_flag.sum()/len(gas_flag)*100:.1f}%)")
        
        return {
            'phi_density': phi_density,
            'phi_neutron': phi_neutron,
            'phi_combined': phi_combined,
            'phi_corrected': phi_corrected,
            'gas_flag': gas_flag
        }
    
    def _calculate_water_saturation(self, porosity_data):
        """Calculate water saturation using Archie's equation."""
        if 'RTAT' not in self.df.columns:
            return None
        
        rt = self.df['RTAT']  # True resistivity
        phi = porosity_data['phi_corrected']
        
        # Archie parameters (typical values, should be calibrated)
        rw = 0.1      # Formation water resistivity (ohm-m) - needs calibration
        a = 1.0       # Tortuosity factor
        m = 2.0       # Cementation exponent
        n = 2.0       # Saturation exponent
        
        # Calculate formation factor
        F = a / (phi ** m)
        
        # Water saturation
        sw = ((rw * F) / rt) ** (1/n)
        sw = sw.clip(0, 1)  # Physical limits
        
        # Hydrocarbon saturation
        sh = 1 - sw
        
        print(f"\nWater Saturation Analysis:")
        print(f"  Formation water resistivity (Rw): {rw} ohm-m")
        print(f"  Archie parameters: a={a}, m={m}, n={n}")
        print(f"  Average water saturation: {sw.mean():.3f} ({sw.mean()*100:.1f}%)")
        print(f"  Average hydrocarbon saturation: {sh.mean():.3f} ({sh.mean()*100:.1f}%)")
        
        return {
            'sw': sw,
            'sh': sh,
            'formation_factor': F,
            'archie_params': {'rw': rw, 'a': a, 'm': m, 'n': n}
        }
    
    def _estimate_permeability(self, porosity_data):
        """Estimate permeability using empirical correlations."""
        phi = porosity_data['phi_corrected']
        
        # Kozeny-Carman equation (modified)
        k_kozeny = 5000 * (phi**3) / ((1-phi)**2)  # mD
        
        # Timur correlation
        k_timur = 0.136 * (phi**4.4) / (1 - phi)**2  # mD
        
        # Average of methods
        k_average = (k_kozeny + k_timur) / 2
        
        print(f"\nPermeability Estimation:")
        print(f"  Kozeny-Carman average: {k_kozeny.mean():.1f} mD")
        print(f"  Timur correlation average: {k_timur.mean():.1f} mD")
        print(f"  Combined average: {k_average.mean():.1f} mD")
        
        return {
            'k_kozeny': k_kozeny,
            'k_timur': k_timur,
            'k_average': k_average
        }
    
    def _calculate_net_to_gross(self):
        """Calculate net-to-gross ratio based on gamma ray."""
        if 'GGCE' not in self.df.columns:
            return None
        
        gr = self.df['GGCE']
        gr_cutoff = 75  # GAPI cutoff for net reservoir
        
        net_flag = gr < gr_cutoff
        total_thickness = len(gr) * 0.5  # Assuming 0.5 ft sampling
        net_thickness = net_flag.sum() * 0.5
        
        ntg_ratio = net_thickness / total_thickness if total_thickness > 0 else 0
        
        print(f"\nNet-to-Gross Analysis:")
        print(f"  GR cutoff: {gr_cutoff} GAPI")
        print(f"  Gross thickness: {total_thickness:.1f} ft")
        print(f"  Net thickness: {net_thickness:.1f} ft")
        print(f"  N/G ratio: {ntg_ratio:.3f} ({ntg_ratio*100:.1f}%)")
        
        return {
            'net_flag': net_flag,
            'ntg_ratio': ntg_ratio,
            'net_thickness': net_thickness,
            'gross_thickness': total_thickness
        }
    
    def create_visualizations(self, results=None):
        """Create comprehensive visualization suite."""
        print("\n📈 CREATING VISUALIZATIONS")
        print("="*50)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Figure 1: Well log display
        self._plot_well_logs()
        
        # Figure 2: Cross-plots
        self._plot_crossplots()
        
        # Figure 3: Histograms
        self._plot_histograms()
        
        # Figure 4: Depth track with interpretation
        if results:
            self._plot_interpretation_track(results)
        
        plt.show()
    
    def _plot_well_logs(self):
        """Create standard well log display."""
        # Select key curves
        key_curves = []
        curve_configs = {
            'GGCE': {'color': 'green', 'scale': 'linear'},
            'SPCG': {'color': 'blue', 'scale': 'linear'},
            'RTAT': {'color': 'red', 'scale': 'log'},
            'NPRL': {'color': 'blue', 'scale': 'linear'},
            'DEN': {'color': 'red', 'scale': 'linear'}
        }
        
        available_curves = [c for c in curve_configs.keys() if c in self.df.columns]
        
        if not available_curves:
            print("No key curves available for plotting")
            return
        
        n_tracks = len(available_curves)
        fig, axes = plt.subplots(1, n_tracks, figsize=(3*n_tracks, 12), sharey=True)
        
        if n_tracks == 1:
            axes = [axes]
        
        for i, curve in enumerate(available_curves):
            data = self.df[curve].dropna()
            if len(data) == 0:
                continue
                
            config = curve_configs[curve]
            
            if config['scale'] == 'log':
                axes[i].semilogx(data.values, data.index, color=config['color'], linewidth=0.5)
            else:
                axes[i].plot(data.values, data.index, color=config['color'], linewidth=0.5)
            
            axes[i].set_xlabel(f"{curve}")
            axes[i].grid(True, alpha=0.3)
            axes[i].invert_yaxis()
            
            if i == 0:
                axes[i].set_ylabel("Depth (ft)")
        
        plt.suptitle(f"Well Log Display - Murphy #1", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('/Users/reuben/well/well_logs_display.png', dpi=300, bbox_inches='tight')
        print("✅ Well logs plot saved as 'well_logs_display.png'")
    
    def _plot_crossplots(self):
        """Create cross-plot analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Neutron-Density crossplot
        if 'NPRL' in self.df.columns and 'DEN' in self.df.columns:
            valid_data = self.df[['NPRL', 'DEN']].dropna()
            if len(valid_data) > 0:
                scatter = axes[0,0].scatter(valid_data['NPRL'], valid_data['DEN'], 
                                         c=valid_data.index, cmap='viridis', alpha=0.6, s=10)
                axes[0,0].set_xlabel('Neutron Porosity (%)')
                axes[0,0].set_ylabel('Bulk Density (g/cc)')
                axes[0,0].set_title('Neutron-Density Cross-plot')
                axes[0,0].grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=axes[0,0], label='Depth (ft)')
        
        # PE vs Density
        if 'PDPE' in self.df.columns and 'DEN' in self.df.columns:
            valid_data = self.df[['PDPE', 'DEN']].dropna()
            if len(valid_data) > 0:
                scatter = axes[0,1].scatter(valid_data['PDPE'], valid_data['DEN'], 
                                         c=valid_data.index, cmap='plasma', alpha=0.6, s=10)
                axes[0,1].set_xlabel('PE (b/e)')
                axes[0,1].set_ylabel('Bulk Density (g/cc)')
                axes[0,1].set_title('PE-Density Cross-plot')
                axes[0,1].grid(True, alpha=0.3)
        
        # Resistivity vs Porosity
        if 'RTAT' in self.df.columns and 'NPRL' in self.df.columns:
            valid_data = self.df[['RTAT', 'NPRL']].dropna()
            if len(valid_data) > 0:
                axes[1,0].loglog(valid_data['RTAT'], valid_data['NPRL'], 
                               'o', alpha=0.6, markersize=3)
                axes[1,0].set_xlabel('Resistivity (ohm-m)')
                axes[1,0].set_ylabel('Neutron Porosity (%)')
                axes[1,0].set_title('Resistivity-Porosity Cross-plot')
                axes[1,0].grid(True, alpha=0.3)
        
        # GR vs SP
        if 'GGCE' in self.df.columns and 'SPCG' in self.df.columns:
            valid_data = self.df[['GGCE', 'SPCG']].dropna()
            if len(valid_data) > 0:
                axes[1,1].scatter(valid_data['GGCE'], valid_data['SPCG'], 
                                alpha=0.6, s=10)
                axes[1,1].set_xlabel('Gamma Ray (GAPI)')
                axes[1,1].set_ylabel('SP (mV)')
                axes[1,1].set_title('GR-SP Cross-plot')
                axes[1,1].grid(True, alpha=0.3)
        
        plt.suptitle('Petrophysical Cross-plots', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('/Users/reuben/well/crossplots_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Cross-plots saved as 'crossplots_analysis.png'")
    
    def _plot_histograms(self):
        """Create histogram analysis of key parameters."""
        key_curves = ['GGCE', 'NPRL', 'DEN', 'RTAT']
        available_curves = [c for c in key_curves if c in self.df.columns]
        
        if not available_curves:
            return
        
        n_curves = len(available_curves)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, curve in enumerate(available_curves[:4]):
            data = self.df[curve].dropna()
            if len(data) == 0:
                continue
            
            axes[i].hist(data, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
            axes[i].set_xlabel(f"{curve}")
            axes[i].set_ylabel("Frequency")
            axes[i].set_title(f"{curve} Distribution")
            axes[i].grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = data.mean()
            std_val = data.std()
            axes[i].axvline(mean_val, color='red', linestyle='--', 
                           label=f'Mean: {mean_val:.2f}')
            axes[i].axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7)
            axes[i].axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7)
            axes[i].legend()
        
        plt.suptitle('Log Data Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('/Users/reuben/well/histograms_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Histograms saved as 'histograms_analysis.png'")
    
    def _plot_interpretation_track(self, results):
        """Create interpretation summary track."""
        fig, axes = plt.subplots(1, 4, figsize=(16, 12), sharey=True)
        
        depth = self.df.index
        
        # Track 1: Lithology
        if 'gamma_ray' in results and results['gamma_ray']:
            lith_data = results['gamma_ray']['lithology']
            lith_colors = {'Clean Sandstone': 'yellow', 'Shaly Sandstone': 'orange', 'Shale': 'brown'}
            
            for lith, color in lith_colors.items():
                mask = lith_data == lith
                if mask.any():
                    axes[0].fill_betweenx(lith_data[mask].index, 0, 1, 
                                        color=color, alpha=0.7, label=lith)
        
        axes[0].set_xlim(0, 1)
        axes[0].set_xlabel("Lithology")
        axes[0].set_ylabel("Depth (ft)")
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # Track 2: Porosity
        if 'porosity' in results and results['porosity']:
            phi = results['porosity']['phi_corrected'] * 100
            axes[1].plot(phi, phi.index, 'blue', linewidth=1)
            axes[1].fill_betweenx(phi.index, 0, phi, alpha=0.3, color='blue')
            axes[1].set_xlim(0, max(30, phi.max()))
        
        axes[1].set_xlabel("Porosity (%)")
        axes[1].grid(True, alpha=0.3)
        
        # Track 3: Water Saturation
        if 'water_saturation' in results and results['water_saturation']:
            sw = results['water_saturation']['sw'] * 100
            sh = results['water_saturation']['sh'] * 100
            axes[2].plot(sw, sw.index, 'blue', linewidth=1, label='Sw')
            axes[2].plot(sh, sh.index, 'red', linewidth=1, label='Sh')
            axes[2].fill_betweenx(sw.index, 0, sw, alpha=0.3, color='blue')
            axes[2].fill_betweenx(sh.index, sw, 100, alpha=0.3, color='red')
            axes[2].set_xlim(0, 100)
        
        axes[2].set_xlabel("Saturation (%)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Track 4: Permeability
        if 'permeability' in results and results['permeability']:
            k = results['permeability']['k_average']
            axes[3].semilogx(k, k.index, 'green', linewidth=1)
            axes[3].set_xlim(0.01, max(1000, k.max()))
        
        axes[3].set_xlabel("Permeability (mD)")
        axes[3].grid(True, alpha=0.3)
        
        # Invert y-axis for all tracks
        for ax in axes:
            ax.invert_yaxis()
        
        plt.suptitle('Petrophysical Interpretation Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('/Users/reuben/well/interpretation_summary.png', dpi=300, bbox_inches='tight')
        print("✅ Interpretation summary saved as 'interpretation_summary.png'")
    
    def generate_report(self, results):
        """Generate comprehensive analysis report."""
        print("\n📋 GENERATING ANALYSIS REPORT")
        print("="*50)
        
        report_content = []
        report_content.append("# MURPHY #1 WELL LOG ANALYSIS REPORT")
        report_content.append("=" * 50)
        report_content.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append(f"LAS File: {self.las_file_path}")
        
        # Well Information
        report_content.append("\n## WELL INFORMATION")
        report_content.append("-" * 30)
        if self.well_info:
            for key, value in self.well_info.items():
                report_content.append(f"{key}: {value}")
        
        # Data Summary
        report_content.append("\n## DATA SUMMARY")
        report_content.append("-" * 30)
        report_content.append(f"Depth Range: {self.df.index.min():.1f} - {self.df.index.max():.1f} ft")
        report_content.append(f"Total Points: {len(self.df):,}")
        report_content.append(f"Available Curves: {len(self.df.columns)}")
        
        # Lithology Summary
        if 'gamma_ray' in results and results['gamma_ray']:
            report_content.append("\n## LITHOLOGY ANALYSIS")
            report_content.append("-" * 30)
            lith_dist = results['gamma_ray']['lithology'].value_counts()
            for lith, count in lith_dist.items():
                percentage = (count / lith_dist.sum()) * 100
                report_content.append(f"{lith}: {percentage:.1f}%")
        
        # Petrophysical Summary
        if 'porosity' in results and results['porosity']:
            report_content.append("\n## PETROPHYSICAL PROPERTIES")
            report_content.append("-" * 30)
            phi_avg = results['porosity']['phi_corrected'].mean() * 100
            report_content.append(f"Average Porosity: {phi_avg:.1f}%")
            
            if 'water_saturation' in results and results['water_saturation']:
                sw_avg = results['water_saturation']['sw'].mean() * 100
                report_content.append(f"Average Water Saturation: {sw_avg:.1f}%")
                report_content.append(f"Average Hydrocarbon Saturation: {100-sw_avg:.1f}%")
            
            if 'permeability' in results and results['permeability']:
                k_avg = results['permeability']['k_average'].mean()
                report_content.append(f"Average Permeability: {k_avg:.1f} mD")
        
        # Reservoir Quality
        report_content.append("\n## RESERVOIR QUALITY ASSESSMENT")
        report_content.append("-" * 30)
        
        if 'porosity' in results and results['porosity']:
            phi_data = results['porosity']['phi_corrected'] * 100
            
            excellent = (phi_data > 15).sum()
            good = ((phi_data > 10) & (phi_data <= 15)).sum()
            fair = ((phi_data > 5) & (phi_data <= 10)).sum()
            poor = (phi_data <= 5).sum()
            total = len(phi_data)
            
            report_content.append(f"Excellent (>15%): {excellent/total*100:.1f}%")
            report_content.append(f"Good (10-15%): {good/total*100:.1f}%")
            report_content.append(f"Fair (5-10%): {fair/total*100:.1f}%")
            report_content.append(f"Poor (<5%): {poor/total*100:.1f}%")
        
        # Write report to file
        report_text = "\n".join(report_content)
        with open('/Users/reuben/well/Murphy1_Analysis_Report.txt', 'w') as f:
            f.write(report_text)
        
        print("✅ Analysis report saved as 'Murphy1_Analysis_Report.txt'")
        return report_text

def main():
    """Main execution function."""
    print("🛢️  ADVANCED LAS FILE ANALYSIS")
    print("=" * 60)
    print("State-of-the-art petrophysical interpretation")
    print("Murphy #1 Well - South Johnson Field")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = PetrophysicalAnalyzer('/Users/reuben/well/Murphy 1_MainPass.las')
    
    # Load and process data
    if not analyzer.load_las_file():
        return
    
    # Data quality assessment
    analyzer.data_quality_assessment()
    
    # Lithology identification
    lithology_results = analyzer.lithology_identification()
    
    # Petrophysical calculations
    petrophysical_results = analyzer.petrophysical_calculations()
    
    # Combine results
    all_results = {**lithology_results, **petrophysical_results}
    
    # Create visualizations
    analyzer.create_visualizations(all_results)
    
    # Generate report
    analyzer.generate_report(all_results)
    
    print("\n🎉 ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Generated files:")
    print("  📊 well_logs_display.png")
    print("  📈 crossplots_analysis.png") 
    print("  📊 histograms_analysis.png")
    print("  📋 interpretation_summary.png")
    print("  📝 Murphy1_Analysis_Report.txt")
    print("\nRecommendations for further analysis:")
    print("  • Calibrate Archie parameters with core data")
    print("  • Perform NMR analysis if available")
    print("  • Conduct pressure transient testing")
    print("  • Integrate with seismic data")
    
if __name__ == "__main__":
    main()