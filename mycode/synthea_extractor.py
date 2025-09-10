"""
synthea_extractor.py - Enhanced Synthea FHIR Data Extraction Framework

DESCRIPTION:
    Comprehensive FHIR data extraction framework specifically designed for Synthea-generated
    synthetic patient bundles. This enhanced version extracts detailed medical records from
    FHIR JSON files and structures them into Python dataclasses for clinical trial matching,
    analytics, and LLM processing. Significantly expanded from the original version with
    additional resource types, date-based indexing, and clinical notes extraction.

ENHANCED FEATURES:
    - Extracts 7+ FHIR resource types (Patients, Encounters, Conditions, Medications, 
      Observations, Procedures, Immunizations, DiagnosticReports)
    - Date-based event indexing for temporal analysis
    - Clinical notes extraction from DiagnosticReports with base64 decoding
    - Enhanced encounter reference tracking across all resource types
    - Comprehensive patient timeline reconstruction
    - Backward compatibility with original SyntheaFHIRExtractor interface

CORE DATA STRUCTURES:
    - Patient: Demographics, contact info, calculated age
    - Condition: Medical diagnoses with onset/resolution tracking
    - Medication: Prescriptions with dosage and indication data
    - Observation: Lab values, vital signs, clinical measurements
    - Procedure: Medical procedures with dates and indications
    - Encounter: Healthcare visits with comprehensive context linking
    - Immunization: Vaccination records with dose tracking
    - DiagnosticReport: Clinical reports with embedded narrative text
    - PatientRecord: Complete longitudinal patient history container

CLINICAL TRIAL MATCHING ENHANCEMENTS:
    - Active condition identification for eligibility screening
    - Current medication tracking for contraindication checking  
    - Temporal event sequencing for history requirements
    - Comprehensive medical context for each encounter
    - Clinical narrative extraction for unstructured criteria matching

DATE-BASED EVENT INDEXING:
    - Automatically indexes all medical events by date (YYYY-MM-DD)
    - Enables efficient temporal queries and timeline reconstruction
    - Supports encounter-based and date-based data retrieval patterns
    - Critical for clinical trial eligibility criteria requiring time sequences

CLINICAL NOTES PROCESSING:
    - Extracts narrative text from DiagnosticReport.presentedForm
    - Handles base64 encoded clinical notes automatically
    - Preserves full clinical context for NLP and LLM analysis
    - Essential for processing unstructured eligibility criteria

ENCOUNTER REFERENCE TRACKING:
    - Links all medical events to originating healthcare encounters
    - Enables encounter-specific context reconstruction
    - Supports both encounter-based and chronological data access patterns
    - Maintains referential integrity across the patient record

INPUT FORMAT:
    - Synthea FHIR R4 JSON bundles (*.json)
    - Complete patient medical history in single bundle format
    - Standard FHIR resource structure with Synthea extensions
    - Supports all Synthea disease modules and specialties

OUTPUT CAPABILITIES:
    - Structured Python dataclasses for programmatic access
    - Active/historical condition categorization
    - Current medication status tracking
    - Encounter-specific medical context extraction
    - Date-based event retrieval for temporal analysis
    - Clinical narrative text for unstructured data mining


"""

import json
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Patient:
    """Patient demographics and identifiers"""
    id: str
    name: str
    birth_date: str
    gender: str
    race: str = ""
    ethnicity: str = ""
    marital_status: str = ""
    address: str = ""
    phone: str = ""
    
    @property
    def age(self) -> int:
        """Calculate current age"""
        birth_year = int(self.birth_date.split('-')[0])
        return 2025 - birth_year


@dataclass
class Condition:
    """Medical condition/diagnosis"""
    id: str
    code: str
    display: str
    clinical_status: str
    onset_date: str
    abatement_date: Optional[str] = None
    encounter_ref: Optional[str] = None
    
    @property
    def is_active(self) -> bool:
        return self.clinical_status == 'active' and not self.abatement_date


@dataclass
class Medication:
    """Medication information"""
    id: str
    code: str
    display: str
    status: str
    authored_on: str
    dosage_text: str = ""
    reason_code: str = ""
    reason_display: str = ""
    encounter_ref: Optional[str] = None


@dataclass
class Observation:
    """Clinical observations (vitals, labs, etc.)"""
    id: str
    code: str
    display: str
    value: Any
    unit: str = ""
    effective_date: str = ""
    category: str = ""
    encounter_ref: Optional[str] = None


@dataclass
class Procedure:
    """Medical procedures"""
    id: str
    code: str
    display: str
    performed_date: str
    status: str
    reason_code: str = ""
    reason_display: str = ""
    encounter_ref: Optional[str] = None


@dataclass
class Immunization:
    """Immunization record"""
    id: str
    vaccine_code: str
    vaccine_display: str
    occurrence_date: str
    status: str
    encounter_ref: Optional[str] = None
    dose_number: Optional[int] = None
    series: Optional[str] = None


@dataclass
class DiagnosticReport:
    """Diagnostic report with clinical notes"""
    id: str
    code: str
    display: str
    effective_date: str
    conclusion: str = ""
    presented_form: str = ""  # Contains actual clinical note text
    encounter_ref: Optional[str] = None


@dataclass
class Encounter:
    """Healthcare encounter/visit"""
    id: str
    type_code: str
    type_display: str
    class_code: str  # ambulatory, emergency, inpatient, etc.
    class_display: str
    start_date: str
    end_date: Optional[str]
    reason_code: str = ""
    reason_display: str = ""
    service_provider: str = ""
    
    # Related data for this encounter
    conditions: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    procedures: List[str] = field(default_factory=list)
    immunizations: List[str] = field(default_factory=list)
    diagnostic_reports: List[str] = field(default_factory=list)


@dataclass
class PatientRecord:
    """Complete patient medical record with enhanced data"""
    patient: Patient
    conditions: Dict[str, Condition] = field(default_factory=dict)
    medications: Dict[str, Medication] = field(default_factory=dict)
    observations: Dict[str, Observation] = field(default_factory=dict)
    procedures: Dict[str, Procedure] = field(default_factory=dict)
    encounters: Dict[str, Encounter] = field(default_factory=dict)
    immunizations: Dict[str, Immunization] = field(default_factory=dict)
    diagnostic_reports: Dict[str, DiagnosticReport] = field(default_factory=dict)
    
    # Group all events by date for easier access
    events_by_date: Dict[str, Dict] = field(default_factory=lambda: defaultdict(lambda: {
        'encounters': [],
        'conditions': [],
        'medications': [],
        'observations': [],
        'procedures': [],
        'immunizations': [],
        'diagnostic_reports': []
    }))
    
    def get_active_conditions(self) -> List[Condition]:
        """Get currently active conditions"""
        return [c for c in self.conditions.values() if c.is_active]
    
    def get_current_medications(self) -> List[Medication]:
        """Get currently active medications"""
        return [m for m in self.medications.values() if m.status == 'active']
    
    def get_events_for_date(self, date: str) -> Dict:
        """Get ALL medical events that happened on a specific date"""
        date_key = date[:10]  # Just YYYY-MM-DD
        return self.events_by_date.get(date_key, {
            'encounters': [],
            'conditions': [],
            'medications': [],
            'observations': [],
            'procedures': [],
            'immunizations': [],
            'diagnostic_reports': []
        })
    
    def get_encounter_context(self, encounter_id: str) -> Dict:
        """Get all medical context for a specific encounter (backward compatible)"""
        encounter = self.encounters.get(encounter_id)
        if not encounter:
            return {}
        
        # Get all events for the encounter date
        date_events = self.get_events_for_date(encounter.start_date)
        
        context = {
            'encounter': encounter,
            'conditions': date_events['conditions'],
            'medications': date_events['medications'],
            'observations': date_events['observations'],
            'procedures': date_events['procedures'],
            'immunizations': date_events['immunizations'],
            'diagnostic_reports': date_events['diagnostic_reports'],
            'patient': self.patient,
            'active_problems': self.get_active_conditions(),
            'current_medications': self.get_current_medications()
        }
        
        return context


class EnhancedSyntheaExtractor:
    """Extract comprehensive medical data from Synthea FHIR bundles"""
    
    def __init__(self):
        self.patient_record = None
    
    def extract_patient_record(self, fhir_bundle_path: Path) -> PatientRecord:
        """Extract complete patient record from FHIR bundle"""
        with open(fhir_bundle_path, 'r', encoding='utf-8') as f:
            bundle = json.load(f)
        
        # Initialize record
        record = PatientRecord(patient=None)
        
        # Track encounter relationships
        encounter_refs = defaultdict(lambda: {
            'conditions': [], 'medications': [], 'observations': [], 
            'procedures': [], 'immunizations': [], 'diagnostic_reports': []
        })
        
        # First pass: extract all resources
        for entry in bundle.get('entry', []):
            resource = entry.get('resource', {})
            resource_type = resource.get('resourceType')
            
            if resource_type == 'Patient':
                record.patient = self._extract_patient(resource)
            
            elif resource_type == 'Encounter':
                encounter = self._extract_encounter(resource)
                record.encounters[encounter.id] = encounter
                # Add to date index
                date_key = encounter.start_date[:10]
                record.events_by_date[date_key]['encounters'].append(encounter)
            
            elif resource_type == 'Condition':
                condition = self._extract_condition(resource)
                record.conditions[condition.id] = condition
                # Track encounter reference if exists
                if condition.encounter_ref:
                    enc_id = condition.encounter_ref.split('/')[-1]
                    encounter_refs[enc_id]['conditions'].append(condition.id)
                # Add to date index
                if condition.onset_date:
                    date_key = condition.onset_date[:10]
                    record.events_by_date[date_key]['conditions'].append(condition)
            
            elif resource_type == 'MedicationRequest':
                medication = self._extract_medication(resource)
                record.medications[medication.id] = medication
                # Track encounter reference if exists
                if medication.encounter_ref:
                    enc_id = medication.encounter_ref.split('/')[-1]
                    encounter_refs[enc_id]['medications'].append(medication.id)
                # Add to date index
                if medication.authored_on:
                    date_key = medication.authored_on[:10]
                    record.events_by_date[date_key]['medications'].append(medication)
            
            elif resource_type == 'Observation':
                observation = self._extract_observation(resource)
                record.observations[observation.id] = observation
                # Track encounter reference if exists
                if observation.encounter_ref:
                    enc_id = observation.encounter_ref.split('/')[-1]
                    encounter_refs[enc_id]['observations'].append(observation.id)
                # Add to date index
                if observation.effective_date:
                    date_key = observation.effective_date[:10]
                    record.events_by_date[date_key]['observations'].append(observation)
            
            elif resource_type == 'Procedure':
                procedure = self._extract_procedure(resource)
                record.procedures[procedure.id] = procedure
                # Track encounter reference if exists
                if procedure.encounter_ref:
                    enc_id = procedure.encounter_ref.split('/')[-1]
                    encounter_refs[enc_id]['procedures'].append(procedure.id)
                # Add to date index
                if procedure.performed_date:
                    date_key = procedure.performed_date[:10]
                    record.events_by_date[date_key]['procedures'].append(procedure)
            
            elif resource_type == 'Immunization':
                immunization = self._extract_immunization(resource)
                record.immunizations[immunization.id] = immunization
                # Track encounter reference if exists
                if immunization.encounter_ref:
                    enc_id = immunization.encounter_ref.split('/')[-1]
                    encounter_refs[enc_id]['immunizations'].append(immunization.id)
                # Add to date index
                if immunization.occurrence_date:
                    date_key = immunization.occurrence_date[:10]
                    record.events_by_date[date_key]['immunizations'].append(immunization)
            
            elif resource_type == 'DiagnosticReport':
                report = self._extract_diagnostic_report(resource)
                record.diagnostic_reports[report.id] = report
                # Track encounter reference if exists
                if report.encounter_ref:
                    enc_id = report.encounter_ref.split('/')[-1]
                    encounter_refs[enc_id]['diagnostic_reports'].append(report.id)
                # Add to date index
                if report.effective_date:
                    date_key = report.effective_date[:10]
                    record.events_by_date[date_key]['diagnostic_reports'].append(report)
        
        # Second pass: link encounters with their related resources
        for encounter_id, refs in encounter_refs.items():
            if encounter_id in record.encounters:
                record.encounters[encounter_id].conditions = refs['conditions']
                record.encounters[encounter_id].medications = refs['medications']
                record.encounters[encounter_id].observations = refs['observations']
                record.encounters[encounter_id].procedures = refs['procedures']
                record.encounters[encounter_id].immunizations = refs['immunizations']
                record.encounters[encounter_id].diagnostic_reports = refs['diagnostic_reports']
        
        return record
    
    def _extract_patient(self, resource: Dict) -> Patient:
        """Extract patient demographics"""
        # Name
        name_parts = resource.get('name', [{}])[0]
        given_names = ' '.join(name_parts.get('given', []))
        family_name = name_parts.get('family', '')
        full_name = f"{given_names} {family_name}".strip()
        
        # Address
        address_parts = resource.get('address', [{}])[0]
        address_lines = address_parts.get('line', [])
        city = address_parts.get('city', '')
        state = address_parts.get('state', '')
        postal = address_parts.get('postalCode', '')
        full_address = f"{', '.join(address_lines)}, {city}, {state} {postal}".strip(', ')
        
        # Phone
        phone = ""
        for telecom in resource.get('telecom', []):
            if telecom.get('system') == 'phone':
                phone = telecom.get('value', '')
                break
        
        # Extensions for race and ethnicity
        race = ""
        ethnicity = ""
        for ext in resource.get('extension', []):
            if 'race' in ext.get('url', ''):
                race_exts = ext.get('extension', [])
                for race_ext in race_exts:
                    if race_ext.get('url') == 'text':
                        race = race_ext.get('valueString', '')
                        break
            elif 'ethnicity' in ext.get('url', ''):
                eth_exts = ext.get('extension', [])
                for eth_ext in eth_exts:
                    if eth_ext.get('url') == 'text':
                        ethnicity = eth_ext.get('valueString', '')
                        break
        
        return Patient(
            id=resource.get('id', ''),
            name=full_name,
            birth_date=resource.get('birthDate', ''),
            gender=resource.get('gender', ''),
            race=race,
            ethnicity=ethnicity,
            marital_status=resource.get('maritalStatus', {}).get('text', ''),
            address=full_address,
            phone=phone
        )
    
    def _extract_condition(self, resource: Dict) -> Condition:
        """Extract condition/diagnosis"""
        code_data = resource.get('code', {}).get('coding', [{}])[0]
        
        return Condition(
            id=resource.get('id', ''),
            code=code_data.get('code', ''),
            display=code_data.get('display', ''),
            clinical_status=resource.get('clinicalStatus', {}).get('coding', [{}])[0].get('code', ''),
            onset_date=resource.get('onsetDateTime', ''),
            abatement_date=resource.get('abatementDateTime'),
            encounter_ref=resource.get('encounter', {}).get('reference', '')
        )
    
    def _extract_medication(self, resource: Dict) -> Medication:
        """Extract medication information"""
        med_code = resource.get('medicationCodeableConcept', {}).get('coding', [{}])[0]
        
        # Dosage
        dosage_text = ""
        dosage_list = resource.get('dosageInstruction', [])
        if dosage_list:
            dosage_text = dosage_list[0].get('text', '')
        
        # Reason
        reason_code = ""
        reason_display = ""
        reason_ref = resource.get('reasonReference', [])
        if reason_ref:
            reason_display = "See conditions"
        
        return Medication(
            id=resource.get('id', ''),
            code=med_code.get('code', ''),
            display=med_code.get('display', ''),
            status=resource.get('status', ''),
            authored_on=resource.get('authoredOn', ''),
            dosage_text=dosage_text,
            reason_code=reason_code,
            reason_display=reason_display,
            encounter_ref=resource.get('encounter', {}).get('reference', '')
        )
    
    def _extract_observation(self, resource: Dict) -> Observation:
        """Extract observation (vitals, labs, etc.)"""
        code_data = resource.get('code', {}).get('coding', [{}])[0]
        
        # Value extraction (can be various types)
        value = None
        unit = ""
        if 'valueQuantity' in resource:
            value = resource['valueQuantity'].get('value')
            unit = resource['valueQuantity'].get('unit', '')
        elif 'valueCodeableConcept' in resource:
            value = resource['valueCodeableConcept'].get('text', '')
        elif 'valueString' in resource:
            value = resource['valueString']
        
        # Category
        category = ""
        if 'category' in resource:
            cat_list = resource.get('category', [])
            if cat_list and isinstance(cat_list, list):
                cat_coding = cat_list[0].get('coding', [{}])[0]
                category = cat_coding.get('code', '')
        
        return Observation(
            id=resource.get('id', ''),
            code=code_data.get('code', ''),
            display=code_data.get('display', ''),
            value=value,
            unit=unit,
            effective_date=resource.get('effectiveDateTime', ''),
            category=category,
            encounter_ref=resource.get('encounter', {}).get('reference', '')
        )
    
    def _extract_procedure(self, resource: Dict) -> Procedure:
        """Extract procedure information"""
        code_data = resource.get('code', {}).get('coding', [{}])[0]
        
        # Reason
        reason_code = ""
        reason_display = ""
        reason_ref = resource.get('reasonReference', [])
        if reason_ref:
            reason_display = "See conditions"
        
        performed_date = resource.get('performedDateTime', '')
        if not performed_date and 'performedPeriod' in resource:
            performed_date = resource['performedPeriod'].get('start', '')
        
        return Procedure(
            id=resource.get('id', ''),
            code=code_data.get('code', ''),
            display=code_data.get('display', ''),
            performed_date=performed_date,
            status=resource.get('status', ''),
            reason_code=reason_code,
            reason_display=reason_display,
            encounter_ref=resource.get('encounter', {}).get('reference', '')
        )
    
    def _extract_immunization(self, resource: Dict) -> Immunization:
        """Extract immunization record"""
        vaccine_code = resource.get('vaccineCode', {}).get('coding', [{}])[0]
        vaccine_display = vaccine_code.get('display', '') or resource.get('vaccineCode', {}).get('text', '')
        
        # Extract dose information if available
        dose_number = None
        series = None
        if 'protocolApplied' in resource:
            protocol = resource['protocolApplied'][0] if resource['protocolApplied'] else {}
            dose_number = protocol.get('doseNumberPositiveInt')
            series = protocol.get('series', '')
        
        return Immunization(
            id=resource.get('id', ''),
            vaccine_code=vaccine_code.get('code', ''),
            vaccine_display=vaccine_display,
            occurrence_date=resource.get('occurrenceDateTime', ''),
            status=resource.get('status', ''),
            encounter_ref=resource.get('encounter', {}).get('reference', ''),
            dose_number=dose_number,
            series=series
        )
    
    def _extract_diagnostic_report(self, resource: Dict) -> DiagnosticReport:
        """Extract diagnostic report with clinical notes"""
        code_data = resource.get('code', {}).get('coding', [{}])[0]
        
        # Extract the actual note text if present
        presented_form = ""
        if 'presentedForm' in resource:
            for form in resource.get('presentedForm', []):
                if 'data' in form:
                    # Try to decode base64 if that's what it is
                    try:
                        decoded = base64.b64decode(form['data']).decode('utf-8')
                        presented_form += decoded + "\n"
                    except:
                        # If not base64, just use as is
                        presented_form += form.get('data', '')
        
        # Get effective date
        effective_date = resource.get('effectiveDateTime', '')
        if not effective_date and 'effectivePeriod' in resource:
            effective_date = resource['effectivePeriod'].get('start', '')
        
        return DiagnosticReport(
            id=resource.get('id', ''),
            code=code_data.get('code', ''),
            display=code_data.get('display', ''),
            effective_date=effective_date,
            conclusion=resource.get('conclusion', ''),
            presented_form=presented_form,
            encounter_ref=resource.get('encounter', {}).get('reference', '')
        )
    
    def _extract_encounter(self, resource: Dict) -> Encounter:
        """Extract encounter information"""
        # Type
        type_list = resource.get('type', [])
        if type_list:
            type_coding = type_list[0].get('coding', [{}])[0]
        else:
            type_coding = {}
        
        # Class
        class_data = resource.get('class', {})
        
        # Period
        period = resource.get('period', {})
        
        # Reason
        reason_code = ""
        reason_display = ""
        reason_list = resource.get('reasonCode', [])
        if reason_list:
            reason_coding = reason_list[0].get('coding', [{}])[0]
            reason_code = reason_coding.get('code', '')
            reason_display = reason_coding.get('display', '')
        
        # Service provider
        provider = resource.get('serviceProvider', {}).get('display', '')
        
        return Encounter(
            id=resource.get('id', ''),
            type_code=type_coding.get('code', ''),
            type_display=type_coding.get('display', '') or 'General Encounter',
            class_code=class_data.get('code', ''),
            class_display=class_data.get('display', '') or class_data.get('code', 'ambulatory'),
            start_date=period.get('start', ''),
            end_date=period.get('end'),
            reason_code=reason_code,
            reason_display=reason_display or 'Routine follow-up',
            service_provider=provider
        )


# Backward compatibility alias
SyntheaFHIRExtractor = EnhancedSyntheaExtractor


# Example usage and testing
if __name__ == "__main__":
    extractor = EnhancedSyntheaExtractor()
    
    # Example path - update with your actual path
    sample_file = Path(r"C:\ClinicalTrialDev\Data\Synthetic\Synthea\run_20250702_133107\fhir\Abram53_Carter549_3a9012e9-54fa-334c-9b17-a053791d34de.json")
    
    if sample_file.exists():
        record = extractor.extract_patient_record(sample_file)
        
        print(f"Patient: {record.patient.name}")
        print(f"Age: {record.patient.age}")
        print(f"\nResource Counts:")
        print(f"  Encounters: {len(record.encounters)}")
        print(f"  Conditions: {len(record.conditions)}")
        print(f"  Medications: {len(record.medications)}")
        print(f"  Observations: {len(record.observations)}")
        print(f"  Procedures: {len(record.procedures)}")
        print(f"  Immunizations: {len(record.immunizations)}")
        print(f"  DiagnosticReports: {len(record.diagnostic_reports)}")
        
        # Show first encounter with all related data
        if record.encounters:
            first_enc_id = list(record.encounters.keys())[0]
            first_enc = record.encounters[first_enc_id]
            print(f"\nFirst Encounter: {first_enc.type_display} on {first_enc.start_date[:10]}")
            
            # Get all events for that date
            events = record.get_events_for_date(first_enc.start_date)
            print(f"  Events on this date:")
            for event_type, event_list in events.items():
                if event_list:
                    print(f"    {event_type}: {len(event_list)}")