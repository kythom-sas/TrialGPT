"""
add_notes_to_csv.py - Clinical Notes Integration with Patient Datasets

DESCRIPTION:
    Integrates LLM-generated clinical notes with existing structured patient datasets.
    This script processes clinical notes text files and merges them with CSV datasets containing patient 
    demographics, encounters, conditions, and medications.

FUNCTIONALITY:
    - Loads existing CSV datasets (patients, encounters, conditions, medications)
    - Parses clinical notes text files to extract individual encounters
    - Extracts SOAP note components (Subjective, Objective, Assessment, Plan)
    - Matches clinical notes to encounters by date
    - Enhances patient records with clinical notes statistics
    - Creates comprehensive datasets combining structured and unstructured data
    - Supports multiple text file formats (.txt, .text)
    - Limits processing to first 5 encounters per patient for efficiency (current testing architecture choice)

CLINICAL NOTE PARSING:
    - Extracts patient names from note headers
    - Identifies encounter sequences and dates using regex patterns
    - Parses SOAP format clinical notes (S:, O:, A:, P: sections)
    - Calculates note statistics (length, word count, sections present)
    - Preserves full note text for downstream text analytics

DATA INTEGRATION:
    - Merges notes with encounters by date matching
    - Enhances patient records with notes count and length statistics
    - Creates new enhanced datasets while preserving originals
    - Handles missing or incomplete note data gracefully

INPUT:
    - CSV files: patients.csv, encounters.csv, conditions.csv, medications.csv
    - Clinical notes directory containing text files with LLM-generated notes
    - Text files formatted with encounter headers and SOAP sections

OUTPUT:
    - clinical_notes.csv: Parsed clinical notes with SOAP components
    - encounters_enhanced.csv: Encounters merged with clinical notes
    - patients_enhanced.csv: Patients with clinical notes statistics
    - All datasets saved in both CSV and Parquet formats

USAGE:
    python add_notes_to_csv.py
    
    Script automatically finds existing CSV datasets and clinical notes directory,
    processes all text files, and creates enhanced datasets.

USE CASE:
    Essential for clinical trial matching as it combines:
    - Structured medical data (codes, dates, demographics)
    - Unstructured clinical narratives (symptoms, observations, plans)
    - SOAP note components for targeted text mining
    This enriched dataset enables more accurate trial eligibility determination.

DEPENDENCIES:
    - pandas: DataFrame operations and data merging
    - pathlib: File system operations  
    - typing: Type hints for data structures
    - re: Regular expression parsing for encounter headers
    - glob: File pattern matching
"""

import pandas as pd
import re
from pathlib import Path
from typing import List, Dict
import glob

class GenericNotesProcessor:
    """Process multiple clinical notes text files"""
    
    def __init__(self, csv_dir: Path):
        self.csv_dir = Path(csv_dir)
        self.datasets = {}
        self._load_existing_datasets()
    
    def _load_existing_datasets(self):
        """Load existing CSV files"""
        csv_files = ['patients.csv', 'encounters.csv', 'conditions.csv', 'medications.csv']
        
        for csv_file in csv_files:
            file_path = self.csv_dir / csv_file
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    dataset_name = csv_file.replace('.csv', '')
                    self.datasets[dataset_name] = df
                    print(f"Loaded {dataset_name}: {len(df)} records")
                except Exception as e:
                    print(f"Error loading {csv_file}: {e}")
    
    def parse_clinical_notes_file(self, file_path: Path) -> List[Dict]:
        """Parse a single clinical notes text file"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []
        
        lines = content.strip().split('\n')
        
        # Extract patient name from first line
        patient_name = ""
        for line in lines[:5]:  # Check first few lines
            if line.startswith('Patient:'):
                patient_name = line.replace('Patient:', '').strip()
                break
        
        if not patient_name:
            print(f"Warning: Could not find patient name in {file_path.name}")
            return []
        
        # Extract generation timestamp
        generated_timestamp = ""
        for line in lines[:10]:
            if line.startswith('Generated:'):
                generated_timestamp = line.replace('Generated:', '').strip()
                break
        
        # Find encounters
        encounters = []
        current_encounter = {}
        current_text = []
        
        for line in lines:
            # Look for encounter headers like "ENCOUNTER 1: 2015-09-14"
            encounter_match = re.match(r'ENCOUNTER\s+(\d+):\s*(\d{4}-\d{2}-\d{2})', line)
            
            if encounter_match:
                # Save previous encounter
                if current_encounter:
                    current_encounter['note_text'] = '\n'.join(current_text).strip()
                    encounters.append(current_encounter)
                
                # Start new encounter
                encounter_num = int(encounter_match.group(1))
                encounter_date = encounter_match.group(2)
                
                current_encounter = {
                    'patient_name': patient_name,
                    'encounter_sequence': encounter_num,
                    'encounter_date': encounter_date,
                    'generated_timestamp': generated_timestamp,
                    'source_file': file_path.name
                }
                current_text = []
                
                # Limit to first 5 encounters per patient
                if encounter_num > 5:
                    break
            
            elif line.strip() and not line.startswith(('Patient:', 'Generated:', '=')):
                current_text.append(line.strip())
        
        # Don't forget the last encounter
        if current_encounter and current_encounter.get('encounter_sequence', 0) <= 5:
            current_encounter['note_text'] = '\n'.join(current_text).strip()
            encounters.append(current_encounter)
        
        # Limit to first 5 encounters
        encounters = encounters[:5]
        
        return encounters
    
    def extract_soap_components(self, note_text: str) -> Dict:
        """Extract SOAP components from clinical note"""
        
        components = {
            'subjective': '',
            'objective': '',
            'assessment': '',
            'plan': ''
        }
        
        if not note_text:
            return components
        
        lines = note_text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            
            # Identify SOAP section headers
            if line.startswith('S:'):
                if current_section:
                    components[current_section] = '\n'.join(current_content).strip()
                current_section = 'subjective'
                current_content = [line[2:].strip()]
            elif line.startswith('O:'):
                if current_section:
                    components[current_section] = '\n'.join(current_content).strip()
                current_section = 'objective'
                current_content = [line[2:].strip()]
            elif line.startswith('A:'):
                if current_section:
                    components[current_section] = '\n'.join(current_content).strip()
                current_section = 'assessment'
                current_content = [line[2:].strip()]
            elif line.startswith('P:'):
                if current_section:
                    components[current_section] = '\n'.join(current_content).strip()
                current_section = 'plan'
                current_content = [line[2:].strip()]
            elif current_section and line:
                current_content.append(line)
        
        # Save the last section
        if current_section:
            components[current_section] = '\n'.join(current_content).strip()
        
        return components
    
    def process_all_notes_files(self, notes_directory: Path) -> pd.DataFrame:
        """Process all clinical notes text files in a directory"""
        
        notes_dir = Path(notes_directory)
        if not notes_dir.exists():
            print(f"Notes directory not found: {notes_dir}")
            return pd.DataFrame()
        
        # Find all text files
        text_files = list(notes_dir.glob('*.txt')) + list(notes_dir.glob('*.text'))
        
        if not text_files:
            print(f"No text files found in {notes_dir}")
            return pd.DataFrame()
        
        print(f"Found {len(text_files)} text files to process")
        
        all_notes = []
        
        for file_path in text_files:
            print(f"Processing: {file_path.name}")
            
            encounters = self.parse_clinical_notes_file(file_path)
            
            for encounter in encounters:
                # Extract SOAP components
                soap_components = self.extract_soap_components(encounter.get('note_text', ''))
                
                # Create record
                record = {
                    'patient_name': encounter['patient_name'],
                    'encounter_sequence': encounter['encounter_sequence'],
                    'encounter_date': encounter['encounter_date'],
                    'note_text': encounter.get('note_text', ''),
                    'note_length': len(encounter.get('note_text', '')),
                    'generated_timestamp': encounter.get('generated_timestamp', ''),
                    'source_file': encounter['source_file'],
                    'subjective_section': soap_components['subjective'],
                    'objective_section': soap_components['objective'],
                    'assessment_section': soap_components['assessment'],
                    'plan_section': soap_components['plan']
                }
                
                all_notes.append(record)
        
        if all_notes:
            notes_df = pd.DataFrame(all_notes)
            print(f"Created clinical notes dataset with {len(notes_df)} notes")
            return notes_df
        else:
            return pd.DataFrame()
    
    def merge_with_existing_data(self, notes_df: pd.DataFrame):
        """Merge clinical notes with existing datasets"""
        
        if notes_df.empty:
            print("No notes to merge")
            return
        
        # Add notes as new dataset
        self.datasets['clinical_notes'] = notes_df
        
        # Try to enhance encounters with notes (match by date only)
        if 'encounters' in self.datasets:
            encounters_enhanced = self._merge_encounters_with_notes(notes_df)
            if encounters_enhanced is not None:
                self.datasets['encounters_enhanced'] = encounters_enhanced
        
        # Enhance patients with notes statistics
        if 'patients' in self.datasets:
            patients_enhanced = self._enhance_patients_with_notes(notes_df)
            if patients_enhanced is not None:
                self.datasets['patients_enhanced'] = patients_enhanced
    
    def _merge_encounters_with_notes(self, notes_df: pd.DataFrame):
        """Merge encounters with notes by date"""
        
        try:
            encounters_df = self.datasets['encounters'].copy()
            
            # Extract date portion from start_date (avoid datetime complications)
            encounters_df['encounter_date_str'] = encounters_df['start_date'].astype(str).str[:10]
            
            # Merge by date string
            merged = encounters_df.merge(
                notes_df[['encounter_date', 'note_text', 'subjective_section', 'objective_section', 
                         'assessment_section', 'plan_section', 'note_length']],
                left_on='encounter_date_str',
                right_on='encounter_date',
                how='left',
                suffixes=('', '_from_notes')
            )
            
            print(f"Enhanced {len(merged)} encounters, {merged['note_text'].notna().sum()} with clinical notes")
            return merged
            
        except Exception as e:
            print(f"Error merging encounters with notes: {e}")
            return None
    
    def _enhance_patients_with_notes(self, notes_df: pd.DataFrame):
        """Add notes statistics to patients"""
        
        try:
            patients_df = self.datasets['patients'].copy()
            
            # Calculate notes stats per patient
            notes_stats = notes_df.groupby('patient_name').agg({
                'encounter_sequence': 'count',
                'note_length': ['mean', 'sum'],
                'generated_timestamp': 'first'
            }).round(2)
            
            notes_stats.columns = ['notes_count', 'avg_note_length', 'total_note_length', 'notes_generated_at']
            notes_stats = notes_stats.reset_index()
            
            # Merge with patients by name
            enhanced = patients_df.merge(
                notes_stats,
                left_on='name',
                right_on='patient_name',
                how='left'
            )
            
            print(f"Enhanced {len(enhanced)} patients, {enhanced['notes_count'].notna().sum()} with clinical notes")
            return enhanced
            
        except Exception as e:
            print(f"Error enhancing patients: {e}")
            return None
    
    def save_datasets(self, output_dir: Path = None):
        """Save all datasets"""
        
        if output_dir is None:
            output_dir = self.csv_dir.parent / f"{self.csv_dir.name}_with_notes"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\nSaving datasets to: {output_dir}")
        print("=" * 50)
        
        for name, df in self.datasets.items():
            try:
                csv_file = output_dir / f"{name}.csv"
                df.to_csv(csv_file, index=False)
                
                # Try to save parquet too
                try:
                    parquet_file = output_dir / f"{name}.parquet"
                    df.to_parquet(parquet_file)
                except:
                    pass
                
                print(f"{name:<20}: {len(df):>6,} records -> {csv_file.name}")
                
            except Exception as e:
                print(f"Error saving {name}: {e}")


def main():
    """Main function"""
    
    # Define paths
    base_dir = Path(r"C:\ClinicalTrialDev\Data\Synthetic\Synthea")
    csv_dir = base_dir / "analytics_ready_datasets_simple"
    
    # Directory containing your clinical notes text files
    notes_dir = Path(r"C:\ClinicalTrialDev\Data\Synthetic\Synthea\enriched_notes_simple_v2")
    
    if not csv_dir.exists():
        print(f"CSV directory not found: {csv_dir}")
        return
    
    if not notes_dir.exists():
        print(f"Clinical notes directory not found: {notes_dir}")
        return
    
    # Initialize processor
    processor = GenericNotesProcessor(csv_dir)
    
    # Process all clinical notes files
    notes_df = processor.process_all_notes_files(notes_dir)
    
    if notes_df.empty:
        print("No clinical notes processed")
        return
    
    # Merge with existing data
    processor.merge_with_existing_data(notes_df)
    
    # Save results
    output_dir = base_dir / "analytics_ready_datasets_with_notes"
    processor.save_datasets(output_dir)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    for name, df in processor.datasets.items():
        print(f"{name:<25}: {len(df):>6,} records")


if __name__ == "__main__":
    main()