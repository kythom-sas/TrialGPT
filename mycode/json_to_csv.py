#!/usr/bin/env python3
"""
json_to_csv.py - Synthea FHIR JSON to CSV Converter

DESCRIPTION:
    Converts Synthea-generated FHIR JSON patient bundles into structured CSV datasets
    for clinical trial matching and analytics. This script processes synthetic patient
    data and extracts key medical information into analytics-ready tabular format.

FUNCTIONALITY:
    - Processes FHIR JSON bundles from Synthea synthetic patient generator
    - Extracts patient demographics, medical conditions, medications, and encounters
    - Converts unstructured FHIR data into structured pandas DataFrames
    - Outputs multiple CSV files for different data types (patients, conditions, medications, encounters)
    - Handles large datasets with progress tracking and error reporting
    - Filters out non-patient files (hospitalInformation, practitionerInformation)
    - Provides data quality metrics and processing statistics

INPUT:
    - Directory containing Synthea FHIR JSON files (*.json)
    - Each JSON file represents a complete patient medical history bundle

OUTPUT:
    - patients.csv: Patient demographics and summary statistics
    - conditions.csv: Medical diagnoses and conditions with status
    - medications.csv: Prescribed medications with codes and dates  
    - encounters.csv: Healthcare visits and encounter types
    - Optional parquet files for efficient storage

USAGE:
    python json_to_csv.py
    
    Script automatically finds the most recent Synthea run directory and processes
    all patient JSON files within the /fhir/ subdirectory.

DEPENDENCIES:
    - pandas: DataFrame operations and CSV export
    - pathlib: File system navigation
    - synthea_extractor: Custom FHIR parsing module
    - json: JSON file processing
"""

import pandas as pd
import json
from pathlib import Path
import sys

# Add your existing code path
sys.path.append(r"C:\ClinicalTrialDev\Scripts")
from synthea_extractor import SyntheaFHIRExtractor

def process_patients_simple(fhir_dir):
    """Simple patient processing without complex date handling"""
    
    extractor = SyntheaFHIRExtractor()
    
    # Initialize collectors
    patients_data = []
    conditions_data = []
    medications_data = []
    encounters_data = []
    
    # Filter out non-patient files
    all_files = list(Path(fhir_dir).glob('*.json'))
    patient_files = [f for f in all_files 
                    if not f.name.startswith(('hospitalInformation', 'practitionerInformation'))]
    
    print(f"Found {len(all_files)} total files, processing {len(patient_files)} patient files")
    
    processed_count = 0
    error_count = 0
    
    for i, fhir_file in enumerate(patient_files, 1):
        if i % 50 == 0:
            print(f"Processed {i}/{len(patient_files)} patients")
            
        try:
            record = extractor.extract_patient_record(fhir_file)
            
            # Skip files that don't have patient records
            if not record or not record.patient or not record.patient.id:
                continue
            
            # Patient data (keep it simple)
            patients_data.append({
                'patient_id': record.patient.id,
                'name': record.patient.name,
                'birth_date': record.patient.birth_date,
                'gender': record.patient.gender,
                'race': record.patient.race,
                'age': record.patient.age,
                'total_encounters': len(record.encounters),
                'total_conditions': len(record.conditions),
                'active_conditions': len(record.get_active_conditions()),
                'current_medications': len(record.get_current_medications()),
            })
            
            # Conditions (simple)
            for cond_id, condition in record.conditions.items():
                conditions_data.append({
                    'patient_id': record.patient.id,
                    'condition_id': cond_id,
                    'code': condition.code,
                    'display': condition.display,
                    'clinical_status': condition.clinical_status,
                    'onset_date': condition.onset_date,
                    'is_active': condition.is_active
                })
            
            # Medications (simple)
            for med_id, medication in record.medications.items():
                medications_data.append({
                    'patient_id': record.patient.id,
                    'medication_id': med_id,
                    'code': medication.code,
                    'display': medication.display,
                    'status': medication.status,
                    'authored_on': medication.authored_on,
                })
            
            # Encounters (simple)
            for enc_id, encounter in record.encounters.items():
                encounters_data.append({
                    'patient_id': record.patient.id,
                    'encounter_id': enc_id,
                    'type_display': encounter.type_display,
                    'class_display': encounter.class_display,
                    'start_date': encounter.start_date,
                    'reason_display': encounter.reason_display,
                })
            
            processed_count += 1
            
        except Exception as e:
            error_count += 1
            if error_count <= 10:  # Only show first 10 errors
                print(f"Error processing {fhir_file.name}: {e}")
    
    print(f"Completed processing: {processed_count} successful, {error_count} errors")
    
    # Convert to DataFrames
    datasets = {
        'patients': pd.DataFrame(patients_data),
        'conditions': pd.DataFrame(conditions_data),
        'medications': pd.DataFrame(medications_data),
        'encounters': pd.DataFrame(encounters_data)
    }
    
    return datasets

def save_simple(datasets, output_dir):
    """Save datasets in simple format"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for dataset_name, df in datasets.items():
        if len(df) > 0:
            print(f"Saving {dataset_name}: {len(df)} records")
            
            # Save as CSV
            df.to_csv(output_dir / f"{dataset_name}.csv", index=False)
            
            # Also save as parquet if possible
            try:
                df.to_parquet(output_dir / f"{dataset_name}.parquet")
            except:
                pass
    
    print(f"All datasets saved to {output_dir}")

def main():
    """Simple main function"""
    
    # Define paths
    base_dir = Path(r"C:\ClinicalTrialDev\Data\Synthetic\Synthea")
    
    # Find most recent run
    run_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
    if not run_dirs:
        print("No Synthea run directories found!")
        return
    
    latest_run = sorted(run_dirs)[-1]
    fhir_dir = latest_run / 'fhir'
    
    print(f"Processing data from: {fhir_dir}")
    
    # Process patients
    datasets = process_patients_simple(fhir_dir)
    
    # Save results
    output_dir = base_dir / "analytics_ready_datasets_simple"
    save_simple(datasets, output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for name, df in datasets.items():
        print(f"{name:15}: {len(df):>6,} records")

if __name__ == "__main__":
    main()