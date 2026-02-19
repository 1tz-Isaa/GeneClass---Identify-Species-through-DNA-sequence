"""Map predicted genus labels to closest clinical disease groups.

This is for research/triage display in the app, not a diagnostic decision.
"""

from __future__ import annotations

import re
from typing import Dict, List


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


# Canonical mapping: genus -> clinical context.
GENUS_CLINICAL_MAP: Dict[str, Dict[str, str]] = {
    # DNA Bacteria
    "streptococcus": {
        "genus": "Streptococcus",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Community-Acquired Pneumonia (CAP)",
        "syndrome_hint": "Typical bacterial CAP",
    },
    "haemophilus": {
        "genus": "Haemophilus",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Community-Acquired Pneumonia (CAP)",
        "syndrome_hint": "Typical bacterial CAP",
    },
    "moraxella": {
        "genus": "Moraxella",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Community-Acquired Pneumonia (CAP)",
        "syndrome_hint": "Airway-associated CAP exacerbation",
    },
    "staphylococcus": {
        "genus": "Staphylococcus",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Community-Acquired Pneumonia (CAP)",
        "syndrome_hint": "Severe necrotizing/secondary bacterial pneumonia",
    },
    "legionella": {
        "genus": "Legionella",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Community-Acquired Pneumonia (CAP)",
        "syndrome_hint": "Atypical CAP (Legionnaires-like)",
    },
    "mycoplasma": {
        "genus": "Mycoplasma",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Community-Acquired Pneumonia (CAP)",
        "syndrome_hint": "Atypical CAP",
    },
    "chlamydia": {
        "genus": "Chlamydia",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Community-Acquired Pneumonia (CAP)",
        "syndrome_hint": "Atypical CAP",
    },
    "klebsiella": {
        "genus": "Klebsiella",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Community-Acquired Pneumonia (CAP)",
        "syndrome_hint": "Lobar/aspiration-associated severe CAP",
    },
    "pseudomonas": {
        "genus": "Pseudomonas",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Hospital-Acquired/Ventilator-Associated Pneumonia (HAP/VAP)",
        "syndrome_hint": "Healthcare-associated resistant pneumonia",
    },
    "acinetobacter": {
        "genus": "Acinetobacter",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Hospital-Acquired/Ventilator-Associated Pneumonia (HAP/VAP)",
        "syndrome_hint": "ICU ventilator-associated pneumonia",
    },
    "serratia": {
        "genus": "Serratia",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Hospital-Acquired/Ventilator-Associated Pneumonia (HAP/VAP)",
        "syndrome_hint": "Nosocomial gram-negative pneumonia",
    },
    "proteus": {
        "genus": "Proteus",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Hospital-Acquired/Ventilator-Associated Pneumonia (HAP/VAP)",
        "syndrome_hint": "Nosocomial gram-negative pneumonia",
    },
    "citrobacter": {
        "genus": "Citrobacter",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Hospital-Acquired/Ventilator-Associated Pneumonia (HAP/VAP)",
        "syndrome_hint": "Nosocomial gram-negative pneumonia",
    },
    "morganella": {
        "genus": "Morganella",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Hospital-Acquired/Ventilator-Associated Pneumonia (HAP/VAP)",
        "syndrome_hint": "Opportunistic nosocomial pneumonia",
    },
    "providencia": {
        "genus": "Providencia",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Hospital-Acquired/Ventilator-Associated Pneumonia (HAP/VAP)",
        "syndrome_hint": "Opportunistic nosocomial pneumonia",
    },
    "burkholderia": {
        "genus": "Burkholderia",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Hospital-Acquired/Ventilator-Associated Pneumonia (HAP/VAP)",
        "syndrome_hint": "Chronic/opportunistic resistant lung infection",
    },
    "mycobacterium": {
        "genus": "Mycobacterium",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Tuberculosis & Nontuberculous Mycobacteria",
        "syndrome_hint": "Tuberculosis/NTM pulmonary disease",
    },
    "nocardia": {
        "genus": "Nocardia",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Tuberculosis & Nontuberculous Mycobacteria",
        "syndrome_hint": "Chronic cavitary/opportunistic pulmonary nocardiosis",
    },
    "rhodococcus": {
        "genus": "Rhodococcus",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Tuberculosis & Nontuberculous Mycobacteria",
        "syndrome_hint": "Opportunistic cavitary pulmonary infection",
    },
    "bacteroides": {
        "genus": "Bacteroides",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Aspiration & Anaerobic Pneumonia",
        "syndrome_hint": "Aspiration-related anaerobic pneumonia/lung abscess",
    },
    "fusobacterium": {
        "genus": "Fusobacterium",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Aspiration & Anaerobic Pneumonia",
        "syndrome_hint": "Necrotizing aspiration-associated anaerobic infection",
    },
    "prevotella": {
        "genus": "Prevotella",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Aspiration & Anaerobic Pneumonia",
        "syndrome_hint": "Aspiration-related anaerobic infection",
    },
    "coxiella": {
        "genus": "Coxiella",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Zoonotic/Environmental Lung Infections",
        "syndrome_hint": "Atypical zoonotic pneumonia",
    },
    "francisella": {
        "genus": "Francisella",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Zoonotic/Environmental Lung Infections",
        "syndrome_hint": "Severe zoonotic pneumonic disease",
    },
    "yersinia": {
        "genus": "Yersinia",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Zoonotic/Environmental Lung Infections",
        "syndrome_hint": "Zoonotic pulmonary infection",
    },
    "pasteurella": {
        "genus": "Pasteurella",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Zoonotic/Environmental Lung Infections",
        "syndrome_hint": "Animal-exposure respiratory infection",
    },
    "brucella": {
        "genus": "Brucella",
        "pathogen_panel": "DNA Bacteria",
        "disease_group": "Zoonotic/Environmental Lung Infections",
        "syndrome_hint": "Zoonotic systemic disease with pulmonary involvement",
    },
    # DNA Fungi
    "aspergillus": {
        "genus": "Aspergillus",
        "pathogen_panel": "DNA Fungi",
        "disease_group": "Opportunistic Fungi",
        "syndrome_hint": "Invasive/chronic pulmonary aspergillosis",
    },
    "candida": {
        "genus": "Candida",
        "pathogen_panel": "DNA Fungi",
        "disease_group": "Opportunistic Fungi",
        "syndrome_hint": "Opportunistic fungal airway/lung colonization or infection",
    },
    "cryptococcus": {
        "genus": "Cryptococcus",
        "pathogen_panel": "DNA Fungi",
        "disease_group": "Opportunistic Fungi",
        "syndrome_hint": "Pulmonary cryptococcosis",
    },
    "pneumocystis": {
        "genus": "Pneumocystis",
        "pathogen_panel": "DNA Fungi",
        "disease_group": "Opportunistic Fungi",
        "syndrome_hint": "Pneumocystis pneumonia (PCP-like)",
    },
    "mucor": {
        "genus": "Mucor",
        "pathogen_panel": "DNA Fungi",
        "disease_group": "Opportunistic Fungi",
        "syndrome_hint": "Pulmonary mucormycosis",
    },
    "histoplasma": {
        "genus": "Histoplasma",
        "pathogen_panel": "DNA Fungi",
        "disease_group": "Endemic Fungi (Geographic)",
        "syndrome_hint": "Endemic fungal pneumonia",
    },
    "coccidioides": {
        "genus": "Coccidioides",
        "pathogen_panel": "DNA Fungi",
        "disease_group": "Endemic Fungi (Geographic)",
        "syndrome_hint": "Valley fever-like fungal pneumonia",
    },
    "blastomyces": {
        "genus": "Blastomyces",
        "pathogen_panel": "DNA Fungi",
        "disease_group": "Endemic Fungi (Geographic)",
        "syndrome_hint": "Endemic blastomycosis pulmonary disease",
    },
    "paracoccidioides": {
        "genus": "Paracoccidioides",
        "pathogen_panel": "DNA Fungi",
        "disease_group": "Endemic Fungi (Geographic)",
        "syndrome_hint": "Chronic endemic fungal pulmonary disease",
    },
    "talaromyces": {
        "genus": "Talaromyces",
        "pathogen_panel": "DNA Fungi",
        "disease_group": "Endemic Fungi (Geographic)",
        "syndrome_hint": "Endemic/opportunistic talaromycosis with lung involvement",
    },
    # RNA Viruses
    "influenzavirus": {
        "genus": "Influenzavirus",
        "pathogen_panel": "RNA Viruses",
        "disease_group": "Orthomyxoviridae",
        "syndrome_hint": "Influenza-like illness / viral pneumonia",
    },
    "betacoronavirus": {
        "genus": "Betacoronavirus",
        "pathogen_panel": "RNA Viruses",
        "disease_group": "Coronaviridae",
        "syndrome_hint": "Coronavirus respiratory disease (SARS-like/COVID-like)",
    },
    "alphacoronavirus": {
        "genus": "Alphacoronavirus",
        "pathogen_panel": "RNA Viruses",
        "disease_group": "Coronaviridae",
        "syndrome_hint": "Coronavirus upper/lower respiratory disease",
    },
    "orthopneumovirus": {
        "genus": "Orthopneumovirus",
        "pathogen_panel": "RNA Viruses",
        "disease_group": "Paramyxoviridae/Pneumoviridae",
        "syndrome_hint": "RSV-like bronchiolitis/viral pneumonia",
    },
    "respirovirus": {
        "genus": "Respirovirus",
        "pathogen_panel": "RNA Viruses",
        "disease_group": "Paramyxoviridae/Pneumoviridae",
        "syndrome_hint": "Parainfluenza-like respiratory infection",
    },
    "metapneumovirus": {
        "genus": "Metapneumovirus",
        "pathogen_panel": "RNA Viruses",
        "disease_group": "Paramyxoviridae/Pneumoviridae",
        "syndrome_hint": "hMPV-like viral lower respiratory infection",
    },
    "enterovirus": {
        "genus": "Enterovirus",
        "pathogen_panel": "RNA Viruses",
        "disease_group": "Picornaviridae",
        "syndrome_hint": "Enteroviral respiratory syndrome",
    },
    "rhinovirus": {
        "genus": "Rhinovirus",
        "pathogen_panel": "RNA Viruses",
        "disease_group": "Picornaviridae",
        "syndrome_hint": "Common-cold like upper respiratory infection",
    },
    "arenavirus": {
        "genus": "Arenavirus",
        "pathogen_panel": "RNA Viruses",
        "disease_group": "Other RNA Respiratory Viruses",
        "syndrome_hint": "Zoonotic viral febrile respiratory syndrome",
    },
    "hantavirus": {
        "genus": "Hantavirus",
        "pathogen_panel": "RNA Viruses",
        "disease_group": "Other RNA Respiratory Viruses",
        "syndrome_hint": "Hantavirus pulmonary syndrome-like disease",
    },
    "orthobunyavirus": {
        "genus": "Orthobunyavirus",
        "pathogen_panel": "RNA Viruses",
        "disease_group": "Other RNA Respiratory Viruses",
        "syndrome_hint": "Arboviral febrile respiratory/neuro-respiratory syndrome",
    },
    "reovirus": {
        "genus": "Reovirus",
        "pathogen_panel": "RNA Viruses",
        "disease_group": "Other RNA Respiratory Viruses",
        "syndrome_hint": "Reoviral respiratory disease",
    },
}


# Extra aliases for genus-like strings that may appear in labels.
GENUS_ALIASES = {
    "bunyavirus": "orthobunyavirus",
    "orthobunya virus": "orthobunyavirus",
    "respiratory syncytial virus": "orthopneumovirus",
    "rsv": "orthopneumovirus",
    "human orthopneumovirus": "orthopneumovirus",
    "human rhinovirus": "rhinovirus",
}


def resolve_genus_key(label: str) -> str | None:
    raw = _normalize(label)
    if not raw:
        return None

    if raw in GENUS_ALIASES:
        raw = GENUS_ALIASES[raw]
    if raw in GENUS_CLINICAL_MAP:
        return raw

    first = raw.split()[0]
    if first in GENUS_ALIASES:
        first = GENUS_ALIASES[first]
    if first in GENUS_CLINICAL_MAP:
        return first

    # Fallback: match any known genus token contained in label.
    for key in sorted(GENUS_CLINICAL_MAP, key=len, reverse=True):
        if key in raw:
            return key

    return None


def infer_closest_disease(prediction: str, top_items: List[Dict[str, float]] | None = None) -> Dict[str, str]:
    key = resolve_genus_key(prediction)

    if key is None and top_items:
        for row in top_items:
            key = resolve_genus_key(str(row.get("label", "")))
            if key is not None:
                break

    if key is None:
        return {
            "matched": "0",
            "genus": "",
            "pathogen_panel": "Unknown",
            "disease_group": "Unknown",
            "syndrome_hint": "No mapping found",
        }

    item = GENUS_CLINICAL_MAP[key]
    return {
        "matched": "1",
        "genus": item["genus"],
        "pathogen_panel": item["pathogen_panel"],
        "disease_group": item["disease_group"],
        "syndrome_hint": item["syndrome_hint"],
    }


def rank_disease_groups(top_items: List[Dict[str, float]], top_n: int = 3) -> List[Dict[str, str]]:
    scores: Dict[str, float] = {}
    support: Dict[str, List[str]] = {}

    for row in top_items or []:
        label = str(row.get("label", ""))
        score = float(row.get("score", 0.0) or 0.0)
        key = resolve_genus_key(label)
        if key is None:
            continue

        info = GENUS_CLINICAL_MAP[key]
        group_key = f"{info['pathogen_panel']} | {info['disease_group']}"
        scores[group_key] = scores.get(group_key, 0.0) + score

        support.setdefault(group_key, [])
        if info["genus"] not in support[group_key]:
            support[group_key].append(info["genus"])

    if not scores:
        return []

    total = sum(scores.values()) or 1.0
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: max(1, top_n)]

    out: List[Dict[str, str]] = []
    for group_key, raw_score in ranked:
        panel, group = group_key.split(" | ", 1)
        out.append(
            {
                "pathogen_panel": panel,
                "disease_group": group,
                "score": str(raw_score / total),
                "supporting_genera": ", ".join(support.get(group_key, [])),
            }
        )
    return out
