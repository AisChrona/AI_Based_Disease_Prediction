import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

class MetadataAnalyzer:
    def __init__(self, metadata_path='data/HAM10000/HAM10000_metadata.csv'):
        self.df = pd.read_csv(metadata_path)
        self.analysis_results = {}
        self.visualization_dir = 'app/static/images/analysis/'
        Path(self.visualization_dir).mkdir(parents=True, exist_ok=True)

    def analyze(self):
        """Run complete analysis pipeline"""
        self._basic_stats()
        self._diagnosis_analysis()
        self._demographic_analysis()
        self._localization_analysis()
        return self.analysis_results

    def _basic_stats(self):
        """Calculate basic dataset statistics"""
        self.analysis_results['total_records'] = len(self.df)
        self.analysis_results['missing_values'] = self.df.isnull().sum().to_dict()

    def _diagnosis_analysis(self):
        """Analyze diagnosis distribution"""
        dx_counts = self.df['dx'].value_counts().to_dict()
        self.analysis_results['diagnosis_distribution'] = dx_counts
        
        # Create visualization
        plt.figure(figsize=(10,6))
        sns.countplot(data=self.df, x='dx', order=dx_counts.keys())
        plt.title('Diagnosis Distribution')
        plt.savefig(f'{self.visualization_dir}diagnosis_distribution.png')
        plt.close()

    def _demographic_analysis(self):
        """Analyze age and sex distributions"""
        # Age analysis
        age_stats = self.df['age'].describe().to_dict()
        self.analysis_results['age_statistics'] = age_stats
        
        plt.figure(figsize=(10,6))
        sns.histplot(data=self.df, x='age', bins=20)
        plt.title('Age Distribution')
        plt.savefig(f'{self.visualization_dir}age_distribution.png')
        plt.close()

        # Sex analysis
        sex_counts = self.df['sex'].value_counts().to_dict()
        self.analysis_results['sex_distribution'] = sex_counts

    def _localization_analysis(self):
        """Analyze lesion localization patterns"""
        loc_counts = self.df['localization'].value_counts().to_dict()
        self.analysis_results['localization_distribution'] = loc_counts

        # Top 10 localizations visualization
        plt.figure(figsize=(12,6))
        sns.countplot(
            data=self.df, 
            y='localization', 
            order=self.df['localization'].value_counts().index[:10]
        )
        plt.title('Top 10 Localization Sites')
        plt.savefig(f'{self.visualization_dir}localization_distribution.png')
        plt.close()

    def save_report(self, output_path='reports/metadata_analysis.json'):
        """Save analysis results to JSON file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2)

if __name__ == '__main__':
    analyzer = MetadataAnalyzer()
    results = analyzer.analyze()
    analyzer.save_report()
