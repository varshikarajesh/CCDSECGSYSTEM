"""
SCP to Clinical Family Mapping Module.
Maps 71 PTB-XL SCP statements into 10 broader clinical families:
1. Normal
2. Rhythm
3. Conduction
4. Ventricular Arrhythmia
5. Hypertrophy
6. Repolarization
7. Ischemia
8. Infarction
9. Pacing
10. Other
"""
import os
import logging
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)

CLINICAL_FAMILIES_LIST = [
    "Normal",
    "Rhythm",
    "Conduction",
    "Ventricular Arrhythmia",
    "Hypertrophy",
    "Repolarization",
    "Ischemia",
    "Infarction",
    "Pacing",
    "Other"
]

# Explicit SCP Code -> Clinical Family Mapping
SCP_TO_FAMILY_RAW: Dict[str, str] = {
    # Normal
    "NORM": "Normal",
    
    # Rhythm (Supraventricular & Sinus)
    "SR": "Rhythm",
    "AFIB": "Rhythm",
    "STACH": "Rhythm",
    "SARRH": "Rhythm",
    "SBRAD": "Rhythm",
    "SVARR": "Rhythm",
    "BIGU": "Rhythm",
    "AFLT": "Rhythm",
    "SVTAC": "Rhythm",
    "PSVT": "Rhythm",
    "TRIGU": "Rhythm",
    
    # Conduction
    "LAFB": "Conduction",
    "IRBBB": "Conduction",
    "1AVB": "Conduction",
    "IVCD": "Conduction",
    "CRBBB": "Conduction",
    "CLBBB": "Conduction",
    "LPFB": "Conduction",
    "WPW": "Conduction",
    "ILBBB": "Conduction",
    "3AVB": "Conduction",
    "2AVB": "Conduction",
    
    # Ventricular Arrhythmia
    "PVC": "Ventricular Arrhythmia",
    "PRC(S)": "Ventricular Arrhythmia",
    
    # Hypertrophy & Atrial Overload
    "LVH": "Hypertrophy",
    "LAO/LAE": "Hypertrophy",
    "RVH": "Hypertrophy",
    "RAO/RAE": "Hypertrophy",
    "SEHYP": "Hypertrophy",
    "VCLVH": "Hypertrophy",
    
    # Repolarization & Waveform Alterations
    "NDT": "Repolarization",
    "NST_": "Repolarization",
    "DIG": "Repolarization",
    "LNGQT": "Repolarization",
    "STD_": "Repolarization",
    "LOWT": "Repolarization",
    "NT_": "Repolarization",
    "INVT": "Repolarization",
    "LVOLT": "Repolarization",
    "HVOLT": "Repolarization",
    "TAB_": "Repolarization",
    "STE_": "Repolarization",
    "EL": "Repolarization",
    
    # Ischemia
    "ISC_": "Ischemia",
    "ISCAL": "Ischemia",
    "ISCIN": "Ischemia",
    "ISCIL": "Ischemia",
    "ISCAS": "Ischemia",
    "ISCLA": "Ischemia",
    "ISCAN": "Ischemia",
    
    # Infarction
    "IMI": "Infarction",
    "ASMI": "Infarction",
    "ILMI": "Infarction",
    "AMI": "Infarction",
    "ALMI": "Infarction",
    "INJAS": "Infarction",
    "LMI": "Infarction",
    "INJAL": "Infarction",
    "IPLMI": "Infarction",
    "IPMI": "Infarction",
    "INJIN": "Infarction",
    "INJLA": "Infarction",
    "PMI": "Infarction",
    "INJIL": "Infarction",
    "QWAVE": "Infarction",
    "ANEUR": "Infarction",
    
    # Pacing
    "PACE": "Pacing",
    
    # Other
    "ABQRS": "Other",
    "PAC": "Other",
    "LPR": "Other",
}

SCP_TO_FAMILY = SCP_TO_FAMILY_RAW


def map_scp_to_family(code: str) -> str:
    """Maps an SCP statement code to its canonical clinical family name."""
    return SCP_TO_FAMILY_RAW.get(str(code).strip(), "Other")


def validate_scp_family_mapping(scp_codes: Optional[List[str]] = None) -> bool:
    """
    Startup assertion helper verifying:
    - NORM resolves to Normal
    - Every provided SCP label maps explicitly to a primary family
    - No silent fallback to Other is allowed unless explicitly mapped
    """
    assert SCP_TO_FAMILY_RAW.get("NORM") == "Normal", "Startup Assertion Failed: NORM must map to 'Normal'!"
    
    if scp_codes:
        for code in scp_codes:
            assert code in SCP_TO_FAMILY_RAW, f"Startup Assertion Failed: SCP code '{code}' not found in SCP_TO_FAMILY_RAW!"
            mapped_fam = SCP_TO_FAMILY_RAW[code]
            assert mapped_fam in CLINICAL_FAMILIES_LIST, f"Startup Assertion Failed: Invalid family '{mapped_fam}' for SCP '{code}'!"
    return True


class SCPFamilyMapper:
    """
    Standalone mapper providing:
    - scp_label -> family_id -> family_name
    - family_id -> List[scp_labels]
    """
    def __init__(self, scp_codes: List[str]):
        self.families = CLINICAL_FAMILIES_LIST
        self.family_to_id = {fam: i for i, fam in enumerate(self.families)}
        self.id_to_family = {i: fam for i, fam in enumerate(self.families)}
        self.scp_codes = scp_codes
        
        # Enforce startup assertion
        validate_scp_family_mapping(scp_codes)
        
        self.scp_to_family_name: Dict[str, str] = {}
        self.scp_to_family_id: Dict[str, int] = {}
        self.family_id_to_scps: Dict[int, List[str]] = {i: [] for i in range(len(self.families))}
        
        for scp in scp_codes:
            fam_name = SCP_TO_FAMILY_RAW[scp]
            fam_id = self.family_to_id[fam_name]
            
            self.scp_to_family_name[scp] = fam_name
            self.scp_to_family_id[scp] = fam_id
            self.family_id_to_scps[fam_id].append(scp)
            
        logger.info(f"Initialized SCPFamilyMapper across {len(self.families)} clinical families for {len(scp_codes)} SCP codes.")

    def get_family_name(self, scp_code: str) -> str:
        if scp_code in self.scp_to_family_name:
            return self.scp_to_family_name[scp_code]
        assert scp_code in SCP_TO_FAMILY_RAW, f"Assertion Failed: Unknown SCP code '{scp_code}' requested!"
        return SCP_TO_FAMILY_RAW[scp_code]

    def get_family_id(self, scp_code: str) -> int:
        fam_name = self.get_family_name(scp_code)
        return self.family_to_id[fam_name]

    def get_scps_in_family(self, family_id: int) -> List[str]:
        return self.family_id_to_scps.get(family_id, [])

    def export_mapping_summary(self) -> pd.DataFrame:
        rows = []
        for scp in self.scp_codes:
            fam_name = self.get_family_name(scp)
            fam_id = self.get_family_id(scp)
            rows.append({
                "SCP Code": scp,
                "Family ID": fam_id,
                "Family Name": fam_name
            })
        return pd.DataFrame(rows)

