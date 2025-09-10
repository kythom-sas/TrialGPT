"""
======================================================================
 Simplified Clinical Note Generator
======================================================================
 Author: Kyle Thomas
======================================================================
 Description:
   A streamlined pipeline to process Synthea FHIR patient bundles and
   generate structured SOAP-style clinical notes using Azure OpenAI.

 Features:
   - Reads patient records from INPUT_DIR
   - Extracts encounters, immunizations, and other events
   - Builds concise prompts for LLM-based SOAP note generation
   - Handles API calls with error handling & token usage reporting
   - Saves enriched notes as both JSON (structured) and TXT (readable)
   - Provides summary statistics (patients processed, errors, cost est.)

 Usage:
   python script.py [number_of_patients]

 Notes:
   - If no limit is given, will prompt confirmation if >100 patients found
   - Requires configuration in config_secret.py
   - Default deployment supports GPT-4/5 or o-series models
======================================================================
"""


import json
import sys
import time
from pathlib import Path
from datetime import datetime
from openai import AzureOpenAI
from synthea_extractor import SyntheaFHIRExtractor
from config_secret import AZURE_ENDPOINT, API_KEY, DEPLOYMENT_NAME, API_VERSION

# Simple configuration
INPUT_DIR = Path(r"C:\ClinicalTrialDev\Data\Synthetic\Synthea\run_20250702_133107\fhir")
OUTPUT_DIR = Path(r"C:\ClinicalTrialDev\Data\Synthetic\Synthea\enriched_notes_simple_v2")
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize
extractor = SyntheaFHIRExtractor()
client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=API_KEY,
    api_version=API_VERSION
)

def process_patient(patient_file: Path):
    """Process patient with enhanced error handling"""
    print(f"\n{'='*60}")
    print(f"Processing: {patient_file.name}")
    
    try:
        # Extract patient data
        record = extractor.extract_patient_record(patient_file)
        patient_name = record.patient.name
        patient_id = record.patient.id
        birth_date = record.patient.birth_date
        
        # Sort encounters by date
        encounters = sorted(record.encounters.items(), key=lambda x: x[1].start_date)
        
        print(f"  Patient: {patient_name}")
        print(f"  Total encounters: {len(encounters)}")
        
        if len(encounters) == 0:
            print("  WARNING: No encounters found")
            return None
            
        # Build prompt (limiting encounters for testing)
        num_encounters = min(5, len(encounters))
        print(f"  Processing first {num_encounters} encounters...")
        
        prompt = f"""Generate clinical SOAP notes for this patient's encounters.

PATIENT: {patient_name}
BIRTH DATE: {birth_date[:10]}

MEDICAL RECORD BY DATE:
"""
        
        for i, (enc_id, encounter) in enumerate(encounters[:num_encounters], 1):
            enc_date = encounter.start_date[:10]
            enc_type = encounter.type_display
            events = record.get_events_for_date(enc_date)
            
            prompt += f"\n{i}. DATE: {enc_date}, ENCOUNTER TYPE: {enc_type}"
            
            # Add immunizations
            if events['immunizations']:
                vacc_list = [f"{imm.vaccine_display}" for imm in events['immunizations']]
                prompt += f"\n   IMMUNIZATIONS ADMINISTERED: {', '.join(vacc_list)}"
            
            # Add other events...
            prompt += "\n"
        
        prompt += """
Generate a SOAP note for each encounter above. Format:

ENCOUNTER 1: [date]
S: [subjective]
O: [objective]
A: [assessment]
P: [plan]

Continue for all encounters.
"""
        
        print(f"  Prompt length: {len(prompt)} characters")
        
        # Call API with explicit error handling
        print(f"  Calling API...")
        
        try:
            # Try the API call
            if DEPLOYMENT_NAME == "o4-mini":
                response = client.chat.completions.create(
                    model=DEPLOYMENT_NAME,
                    messages=[
                        {"role": "system", "content": "You are a physician writing clinical notes."},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=2000,
                    timeout=30  # Add timeout
                )
            else:
                response = client.chat.completions.create(
                    model=DEPLOYMENT_NAME,
                    messages=[
                        {"role": "system", "content": "You are a physician writing clinical notes."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000,
                    timeout=30  # Add timeout
                )
            
            print(f"  API call successful")
            
            # Extract response
            generated_notes = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            print(f"  Generated {len(generated_notes)} chars using {tokens_used} tokens")
            
        except Exception as api_error:
            print(f"  API ERROR: {type(api_error).__name__}: {api_error}")
            return None
        
        # Save results with error handling
        try:
            result = {
                "patient_id": patient_id,
                "patient_name": patient_name,
                "encounters_processed": num_encounters,
                "generated_notes": generated_notes,
                "tokens_used": tokens_used,
                "generated_at": datetime.now().isoformat()
            }
            
            # Save JSON
            output_file = OUTPUT_DIR / f"patient_{patient_id}_notes.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved JSON to {output_file.name}")
            
            # Save text
            text_file = OUTPUT_DIR / f"patient_{patient_id}_notes.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(f"Patient: {patient_name}\n")
                f.write(f"Generated: {datetime.now()}\n")
                f.write("="*60 + "\n\n")
                f.write(generated_notes)
            print(f"  Saved text to {text_file.name}")
            
            return result
            
        except Exception as save_error:
            print(f"  SAVE ERROR: {save_error}")
            return None
        
    except Exception as e:
        print(f"  UNEXPECTED ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Process patient files"""
    
    # Check for command line argument
    max_patients = None
    if len(sys.argv) > 1:
        try:
            max_patients = int(sys.argv[1])
            print(f"Processing limit set to {max_patients} patients")
        except ValueError:
            print(f"Invalid argument '{sys.argv[1]}'. Usage: python script.py [number_of_patients]")
            sys.exit(1)
    
    # Get all patient files
    patient_files = list(INPUT_DIR.glob("*.json"))
    print(f"Found {len(patient_files)} patient files")
    
    # Limit if specified
    if max_patients:
        patient_files = patient_files[:max_patients]
        print(f"Processing first {len(patient_files)} patients")
    
    # Confirm before processing all
    if not max_patients and len(patient_files) > 100:
        response = input(f"Process all {len(patient_files)} patients? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted. Use 'python script.py 10' to process first 10 patients.")
            sys.exit(0)
    
    # Process them
    results = []
    total_tokens = 0
    errors = 0
    
    for i, patient_file in enumerate(patient_files, 1):
        print(f"\nProgress: {i}/{len(patient_files)}")
        
        result = process_patient(patient_file)
        
        if result:
            results.append(result)
            total_tokens += result['tokens_used']
        else:
            errors += 1
        
        # Rate limiting
        time.sleep(0.5)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Processed: {len(results)}/{len(patient_files)} patients")
    print(f"Errors: {errors}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Est. cost: ${(total_tokens/1000)*0.02:.2f}")
    print(f"Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()