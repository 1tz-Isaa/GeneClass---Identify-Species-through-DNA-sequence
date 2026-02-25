import os
import re
import smtplib
import ssl
import io
import csv
import http.client
import time
import socket
import urllib.error
import urllib.request
import zipfile
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord

# =========================
# CONFIG
# =========================
# NCBI yêu cầu email thật
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "bao.luu1702@gmail.com")
NCBI_API_KEY = os.getenv("NCBI_API_KEY")
Entrez.email = NCBI_EMAIL
Entrez.api_key = NCBI_API_KEY
Entrez.max_tries = int(os.getenv("ENTREZ_MAX_TRIES", "3"))
Entrez.sleep_between_tries = int(os.getenv("ENTREZ_SLEEP_BETWEEN_TRIES", "2"))

SAMPLES_PER_SPECIES = max(1, int(os.getenv("SAMPLES_PER_SPECIES", "8")))
DNA_SAMPLES_PER_SPECIES = max(
    1, int(os.getenv("DNA_SAMPLES_PER_SPECIES", os.getenv("SAMPLES_PER_SPECIES_DNA", str(SAMPLES_PER_SPECIES))))
)
RNA_SAMPLES_PER_SPECIES = max(
    1, int(os.getenv("RNA_SAMPLES_PER_SPECIES", os.getenv("SAMPLES_PER_SPECIES_RNA", "20")))
)
RNA_REQUIRE_COMPLETE_GENOME = os.getenv("RNA_REQUIRE_COMPLETE_GENOME", "1") == "1"
SEARCH_RETMAX = 800
FASTA_PARSE_FORMAT = os.getenv("FASTA_PARSE_FORMAT", "fasta-blast")
SKIP_COMPLETED = os.getenv("SKIP_COMPLETED", "1") == "1"
FORCE_REBUILD = os.getenv("FORCE_REBUILD", "0") == "1"
REFILL_ONLY_MISSING = os.getenv("REFILL_ONLY_MISSING", "1") == "1"
AUDIT_ONLY = os.getenv("AUDIT_ONLY", "0") == "1"
AUDIT_CHECK_NCBI = os.getenv("AUDIT_CHECK_NCBI", "0") == "1"
AUDIT_REPORT_PATH = Path(os.getenv("AUDIT_REPORT_PATH", "runs/training_logs/dataset_audit.tsv"))
ESEARCH_RETRIES = int(os.getenv("ESEARCH_RETRIES", "4"))
ESEARCH_BACKOFF_SEC = float(os.getenv("ESEARCH_BACKOFF_SEC", "1.5"))
EFETCH_RETRIES = int(os.getenv("EFETCH_RETRIES", "4"))
EFETCH_BACKOFF_SEC = float(os.getenv("EFETCH_BACKOFF_SEC", "2.0"))
EFETCH_BATCH_SIZE = max(1, int(os.getenv("EFETCH_BATCH_SIZE", "20")))
DATASET_TARGETS = {
    x.strip().lower()
    for x in os.getenv("DATASET_TARGETS", "rna, bacteria, fungi").split(",")
    if x.strip()
}
if not DATASET_TARGETS:
    DATASET_TARGETS = {"rna"}

# Search profiles theo từng nhóm dữ liệu
PROFILE_BACTERIA = {
    "join_genus": True,
    "require_species_phrase": True,
    "min_len": 800,
    "max_len": 2000,
    "title_term": (
        '("16S ribosomal RNA"[Title] OR "16S rRNA"[Title] '
        'OR "small subunit ribosomal RNA"[Title] OR "16S"[Title])'
    ),
    "relaxed_term": (
        '("16S ribosomal RNA"[All Fields] OR "16S rRNA"[All Fields] '
        'OR "small subunit ribosomal RNA"[All Fields] OR "16S"[All Fields])'
    ),
    "exclude_block": (
        '("whole genome shotgun"[Title] OR genome[Title] OR chromosome[Title] '
        'OR contig[Title] OR scaffold[Title])'
    ),
    "include_terms": ["16s", "ribosomal rna", "rrna", "small subunit ribosomal rna"],
    "bad_terms": ["whole genome shotgun", "wgs", "contig", "scaffold", "chromosome"],
}

PROFILE_FUNGI = {
    "join_genus": True,
    "require_species_phrase": True,
    "min_len": 250,
    "max_len": 3500,
    "title_term": (
        '("internal transcribed spacer"[Title] OR "ITS"[Title] '
        'OR "18S ribosomal RNA"[Title] OR "28S ribosomal RNA"[Title])'
    ),
    "relaxed_term": (
        '("internal transcribed spacer"[All Fields] OR "ITS"[All Fields] '
        'OR "18S ribosomal RNA"[All Fields] OR "28S ribosomal RNA"[All Fields])'
    ),
    "exclude_block": (
        '("whole genome shotgun"[Title] OR genome[Title] OR chromosome[Title] '
        'OR contig[Title] OR scaffold[Title])'
    ),
    "include_terms": [
        "internal transcribed spacer",
        "its",
        "its1",
        "its2",
        "5.8s",
        "18s",
        "28s",
        "ribosomal rna",
        "ribosomal dna",
        "rdna",
    ],
    "bad_terms": ["whole genome shotgun", "wgs", "contig", "scaffold", "chromosome"],
}

PROFILE_RNA = {
    "join_genus": False,
    "require_species_phrase": False,
    "min_len": 500,
    "max_len": 40000,
    "title_term": (
        '("complete genome"[Title])'
        if RNA_REQUIRE_COMPLETE_GENOME
        else '(("complete genome"[Title]) OR (segment[Title]) OR (RNA[Title]))'
    ),
    "relaxed_term": (
        '("complete genome"[All Fields])'
        if RNA_REQUIRE_COMPLETE_GENOME
        else '(("complete genome"[All Fields]) OR (segment[All Fields]) OR (RNA[All Fields]))'
    ),
    "exclude_block": (
        '(patent[Title] OR vector[Title] OR "synthetic construct"[Organism] OR "partial sequence"[Title] '
        'OR "partial cds"[Title] OR "complete cds"[Title])'
    ),
    "include_terms": [],
    "bad_terms": [
        "vector",
        "plasmid",
        "synthetic construct",
        "partial sequence",
        "partial cds",
        "complete cds",
    ],
    "use_aliases": True,
    "organism_fallback_all_fields": True,
    "require_complete_genome": RNA_REQUIRE_COMPLETE_GENOME,
}

SEARCH_PROFILES = {
    "bacteria": PROFILE_BACTERIA,
    "fungi": PROFILE_FUNGI,
    "rna": PROFILE_RNA,
}

RNA_ORGANISM_ALIASES: Dict[str, List[str]] = {
    "sars-cov-2": ["Severe acute respiratory syndrome coronavirus 2", "SARS-CoV-2"],
    "sars-cov": [
        "SARS coronavirus",
        "Severe acute respiratory syndrome-related coronavirus",
        "SARS-CoV",
    ],
    "mers-cov": [
        "Middle East respiratory syndrome-related coronavirus",
        "MERS-CoV",
    ],
    "human orthopneumovirus": ["Human orthopneumovirus", "Respiratory syncytial virus", "RSV"],
    "human respirovirus 1": ["Human respirovirus 1", "Human parainfluenza virus 1", "HPIV-1"],
    "human respirovirus 3": ["Human respirovirus 3", "Human parainfluenza virus 3", "HPIV-3"],
    "human metapneumovirus": ["Human metapneumovirus", "hMPV"],
}

# -------------------------
# DNA Bacteria (27 genera)
# -------------------------
BACTERIA_GENUS_TO_SPECIES: Dict[str, List[str]] = {
    "Streptococcus": ["pneumoniae", "pyogenes", "agalactiae", "anginosus", "mitis"],
    "Haemophilus": ["influenzae", "parainfluenzae", "haemolyticus", "ducreyi", "aegyptius"],
    "Moraxella": ["catarrhalis", "osloensis", "nonliquefaciens", "lacunata", "atlantae"],
    "Staphylococcus": ["aureus", "epidermidis", "haemolyticus", "lugdunensis", "saprophyticus"],
    "Legionella": ["pneumophila", "longbeachae", "micdadei", "bozemanii", "dumoffii"],
    "Mycoplasma": ["pneumoniae", "hominis", "genitalium", "fermentans", "penetrans"],
    "Chlamydia": ["pneumoniae", "psittaci", "trachomatis", "abortus", "felis"],
    "Klebsiella": ["pneumoniae", "oxytoca", "variicola", "quasipneumoniae", "michiganensis"],
    "Pseudomonas": ["aeruginosa", "putida", "fluorescens", "stutzeri", "mendocina"],
    "Acinetobacter": ["baumannii", "nosocomialis", "pittii", "lwoffii", "johnsonii"],
    "Serratia": ["marcescens", "liquefaciens", "fonticola", "plymuthica", "rubidaea"],
    "Proteus": ["mirabilis", "vulgaris", "penneri", "hauseri", "terrae"],
    "Citrobacter": ["freundii", "koseri", "braakii", "amalonaticus", "youngae"],
    "Morganella": [
        "morganii",
        "psychrotolerans",
        "morganii subsp. morganii",
        "morganii subsp. sibonii",
        "sp.",
    ],
    "Providencia": ["stuartii", "rettgeri", "alcalifaciens", "rustigianii", "heimbachae"],
    "Burkholderia": ["cepacia", "cenocepacia", "multivorans", "vietnamiensis", "gladioli"],
    "Mycobacterium": ["tuberculosis", "avium", "intracellulare", "kansasii", "abscessus"],
    "Nocardia": ["asteroides", "farcinica", "nova", "brasiliensis", "cyriacigeorgica"],
    "Rhodococcus": ["equi", "erythropolis", "ruber", "opacus", "fascians"],
    "Bacteroides": ["fragilis", "thetaiotaomicron", "vulgatus", "uniformis", "ovatus"],
    "Fusobacterium": ["nucleatum", "necrophorum", "varium", "mortiferum", "periodonticum"],
    "Prevotella": ["melaninogenica", "intermedia", "nigrescens", "denticola", "oris"],
    "Coxiella": [
        "burnetii",
        "cheraxi",
        "sp. tick endosymbiont",
        "sp. crustacean endosymbiont",
        "sp. environmental isolate",
    ],
    "Francisella": ["tularensis", "novicida", "philomiragia", "hispaniensis", "noatunensis"],
    "Yersinia": ["pestis", "enterocolitica", "pseudotuberculosis", "ruckeri", "rohdei"],
    "Pasteurella": ["multocida", "canis", "dagmatis", "stomatis", "pneumotropica"],
    "Brucella": ["melitensis", "abortus", "suis", "canis", "ovis"],
}

# ---------------------
# DNA Fungi (10 genera)
# ---------------------
FUNGI_GENUS_TO_SPECIES: Dict[str, List[str]] = {
    "Aspergillus": ["fumigatus", "flavus", "niger", "terreus", "nidulans"],
    "Candida": ["albicans", "glabrata", "tropicalis", "parapsilosis", "auris"],
    "Cryptococcus": ["neoformans", "gattii", "laurentii", "albidus", "curvatus"],
    "Pneumocystis": ["jirovecii", "carinii", "murina", "wakefieldiae", "oryctolagi"],
    "Mucor": ["circinelloides", "racemosus", "hiemalis", "indicus", "plumbeus"],
    "Histoplasma": ["capsulatum", "duboisii", "ohiense", "mississippiense", "suramericanum"],
    "Coccidioides": [
        "immitis",
        "posadasii",
        "sp. clinical isolate",
        "sp. environmental isolate",
        "sp. genomic assembly",
    ],
    "Blastomyces": ["dermatitidis", "gilchristii", "helicus", "percursus", "silverae"],
    "Paracoccidioides": ["brasiliensis", "lutzii", "americana", "restrepiensis", "venezuelensis"],
    "Talaromyces": ["marneffei", "stipitatus", "purpurogenus", "verruculosus", "islandicus"],
}

# ------------------
# RNA Viruses (12)
# ------------------
RNA_GENUS_TO_SPECIES: Dict[str, List[str]] = {
    "Influenzavirus": [
        "Influenza A virus",
        "Influenza B virus",
        "Influenza C virus",
        "Influenza D virus",
    ],
    "Betacoronavirus": [
        "SARS-CoV-2",
        "SARS-CoV",
        "MERS-CoV",
        "Human coronavirus OC43",
        "Human coronavirus HKU1",
    ],
    "Alphacoronavirus": [
        "Human coronavirus 229E",
        "Human coronavirus NL63",
        "Feline coronavirus",
        "Canine coronavirus",
        "Porcine epidemic diarrhea virus",
    ],
    "Orthopneumovirus": [
        "Human orthopneumovirus",
        "Bovine orthopneumovirus",
        "Murine orthopneumovirus",
        "Ovine orthopneumovirus",
        "Caprine orthopneumovirus",
    ],
    "Respirovirus": [
        "Human respirovirus 1",
        "Human respirovirus 3",
        "Bovine respirovirus 3",
        "Porcine respirovirus 1",
        "Murine respirovirus",
    ],
    "Metapneumovirus": [
        "Human metapneumovirus",
        "Avian metapneumovirus A",
        "Avian metapneumovirus B",
        "Avian metapneumovirus C",
        "Avian metapneumovirus D",
    ],
    "Enterovirus": [
        "Enterovirus D68",
        "Enterovirus A71",
        "Coxsackievirus A21",
        "Poliovirus 1",
        "Echovirus 30",
    ],
    "Rhinovirus": [
        "Human rhinovirus A",
        "Human rhinovirus B",
        "Human rhinovirus C",
    ],
    "Arenavirus": ["Lassa virus", "Junin virus", "Machupo virus", "Guanarito virus", "Sabia virus"],
    "Hantavirus": ["Hantaan virus", "Sin Nombre virus", "Andes virus", "Puumala virus", "Seoul virus"],
    "Orthobunyavirus": [
        "La Crosse virus",
        "Oropouche virus",
        "Bunyamwera virus",
        "Jamestown Canyon virus",
        "California encephalitis virus",
    ],
    "Reovirus": [
        "Mammalian orthoreovirus",
        "Avian orthoreovirus",
        "Nelson Bay orthoreovirus",
        "Baboon orthoreovirus",
        "Reptilian orthoreovirus",
    ],
}

DATASET_COLLECTIONS = [
    {
        "label": "DNA/Bacteria",
        "root": Path("DNA/Bacteria"),
        "profile": "bacteria",
        "genus_to_species": BACTERIA_GENUS_TO_SPECIES,
    },
    {
        "label": "DNA/Fungi",
        "root": Path("DNA/Fungi"),
        "profile": "fungi",
        "genus_to_species": FUNGI_GENUS_TO_SPECIES,
    },
    {
        "label": "RNA/Viruses",
        "root": Path("RNA/Viruses"),
        "profile": "rna",
        "genus_to_species": RNA_GENUS_TO_SPECIES,
    },
]

# Email report (optional)
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
EMAIL_TO = os.getenv("EMAIL_TO")
EMAIL_ATTACH_ZIP = os.getenv("EMAIL_ATTACH_ZIP", "0") == "1"
NCBI_SSL_CA_FILE = os.getenv("NCBI_SSL_CA_FILE")
NCBI_INSECURE_SSL = os.getenv("NCBI_INSECURE_SSL", "0") == "1"


# =========================
# HELPERS
# =========================
def configure_ssl_for_entrez() -> None:
    if NCBI_INSECURE_SSL:
        ctx = ssl._create_unverified_context()
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
        urllib.request.install_opener(opener)
        print("[SSL] Warning: NCBI_INSECURE_SSL=1 -> bỏ verify SSL (không khuyến nghị lâu dài).")
        return

    ca_file = NCBI_SSL_CA_FILE
    if not ca_file:
        try:
            import certifi

            ca_file = certifi.where()
        except Exception:
            ca_file = None

    if ca_file:
        ctx = ssl.create_default_context(cafile=ca_file)
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
        urllib.request.install_opener(opener)
        print(f"[SSL] Using CA bundle: {ca_file}")
    else:
        print("[SSL] Using system certificate store.")


def batched(items: List[str], size: int) -> List[List[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def normalize_token(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def safe_path_name(name: str) -> str:
    name = name.strip().replace("/", "-")
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_.()\-]", "", name)
    return name or "unknown"


def extract_accession(header: str) -> str:
    first = header.split()[0]
    return first.strip("|>")


def extract_strain_key(header: str) -> str:
    patterns = [
        r"\bstrain\s*[:=]?\s*([^,;|]+)",
        r"\bstr\.?\s*([^,;|]+)",
        r"\bisolate\s*[:=]?\s*([^,;|]+)",
        r"\bsubstr\.?\s*([^,;|]+)",
        r"\bsubtype\s*[:=]?\s*([^,;|]+)",
        r"\bvariant\s*[:=]?\s*([^,;|]+)",
    ]

    for pat in patterns:
        m = re.search(pat, header, flags=re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            raw = re.sub(r"\b(contig|scaffold)\b.*$", "", raw, flags=re.IGNORECASE).strip()
            key = normalize_token(raw)
            if key:
                return key

    return normalize_token(extract_accession(header))


def build_organism_name(genus: str, species: str, join_genus: bool) -> str:
    s = " ".join(species.split())
    if not join_genus:
        return s
    if s.lower().startswith(genus.lower() + " "):
        return s
    return f"{genus} {s}"


def normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def target_samples_for_profile(profile_name: str) -> int:
    name = profile_name.strip().lower()
    if name == "rna":
        return int(RNA_SAMPLES_PER_SPECIES)
    return int(DNA_SAMPLES_PER_SPECIES)


def expand_organism_names(base_name: str, use_aliases: bool) -> List[str]:
    names = [base_name]
    if use_aliases:
        alias_key = normalize_key(base_name)
        for k, aliases in RNA_ORGANISM_ALIASES.items():
            if normalize_key(k) == alias_key:
                names.extend(aliases)
                break

    seen = set()
    out = []
    for n in names:
        key = normalize_key(n)
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out


def build_species_queries(genus: str, species: str, profile: Dict[str, object]) -> List[str]:
    organism_name = build_organism_name(genus, species, bool(profile["join_genus"]))
    organism_names = expand_organism_names(
        base_name=organism_name,
        use_aliases=bool(profile.get("use_aliases", False)),
    )

    min_len = int(profile["min_len"])
    max_len = int(profile["max_len"])
    length = f"{min_len}:{max_len}[SLEN]"
    title_term = str(profile["title_term"])
    relaxed_term = str(profile["relaxed_term"])
    exclude_block = str(profile.get("exclude_block", "")).strip()
    use_all_fields_fallback = bool(profile.get("organism_fallback_all_fields", False))

    queries = []

    for name in organism_names:
        organism_fields = [f'"{name}"[Organism]']
        if use_all_fields_fallback:
            organism_fields.append(f'"{name}"[All Fields]')

        for organism in organism_fields:
            if exclude_block:
                queries.append(f"{organism} AND {title_term} AND {length} AND NOT {exclude_block}")
                queries.append(f"{organism} AND {relaxed_term} AND {length} AND NOT {exclude_block}")
            queries.append(f"{organism} AND {relaxed_term} AND {length}")
            queries.append(f"{organism} AND {relaxed_term}")

    # Dedupe query strings, giữ thứ tự.
    deduped = []
    seen_q = set()
    for q in queries:
        if q in seen_q:
            continue
        seen_q.add(q)
        deduped.append(q)

    queries = deduped
    return queries


def is_non_target(header: str, profile: Dict[str, object]) -> bool:
    h = header.lower()
    bad_terms = [str(t).lower() for t in profile.get("bad_terms", [])]
    if any(t in h for t in bad_terms):
        return True

    if bool(profile.get("require_complete_genome", False)):
        has_complete_genome = "complete genome" in h
        has_complete_segment = "segment" in h and "complete sequence" in h
        if not (has_complete_genome or has_complete_segment):
            return True

    include_terms = [str(t).lower() for t in profile.get("include_terms", [])]
    if include_terms and not any(t in h for t in include_terms):
        return True

    return False


def search_ids(genus: str, species: str, profile: Dict[str, object], target_count: int) -> List[str]:
    queries = build_species_queries(genus, species, profile)
    seen = set()
    collected: List[str] = []
    network_failed = False
    retries = max(1, ESEARCH_RETRIES)

    for idx, term in enumerate(queries, start=1):
        rec = None
        for attempt in range(1, retries + 1):
            try:
                with Entrez.esearch(db="nucleotide", term=term, retmax=SEARCH_RETMAX) as handle:
                    rec = Entrez.read(handle)
                break
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                http.client.IncompleteRead,
                RuntimeError,
                ValueError,
                TimeoutError,
                socket.timeout,
                ConnectionError,
                OSError,
            ) as e:
                network_failed = True
                print(f"[WARN] esearch tier{idx} attempt {attempt}/{retries} failed: {e}")
                if attempt < retries:
                    sleep_s = ESEARCH_BACKOFF_SEC * attempt
                    print(f"[WARN] retry tier{idx} after {sleep_s:.1f}s")
                    time.sleep(sleep_s)
            except Exception as e:
                network_failed = True
                print(f"[WARN] esearch tier{idx} unexpected error: {e}")
                break

        if rec is None:
            continue

        ids = rec.get("IdList", [])
        print(f"[NCBI] tier{idx} ids: {len(ids)}")

        for rid in ids:
            if rid in seen:
                continue
            seen.add(rid)
            collected.append(rid)

        if len(collected) >= max(target_count * 12, 120):
            break

    if network_failed and not collected:
        print(f"[WARN] search_ids got no result due to network/API issue for {genus} {species}")

    return collected


def fetch_fasta_records(ids: List[str]) -> List[SeqRecord]:
    if not ids:
        return []

    records: List[SeqRecord] = []

    def parse_fasta_payload(raw_text: str) -> List[SeqRecord]:
        if not raw_text:
            return []

        # Bỏ dòng rỗng và chỉ giữ phần có dạng FASTA/comment hợp lệ.
        lines = [line.rstrip() for line in raw_text.splitlines() if line.strip()]
        if not lines:
            return []

        start_idx = 0
        while start_idx < len(lines) and not lines[start_idx].startswith((">", "#", "!", ";")):
            start_idx += 1
        lines = lines[start_idx:]

        if not lines or not any(line.startswith(">") for line in lines):
            return []

        cleaned = "\n".join(lines) + "\n"

        # Fallback parser để chịu được comment header khác nhau.
        for fmt in (FASTA_PARSE_FORMAT, "fasta-pearson", "fasta"):
            try:
                parsed = list(SeqIO.parse(io.StringIO(cleaned), fmt))
                if parsed:
                    return parsed
            except ValueError:
                continue

        return []

    def to_text(data: object) -> str:
        if data is None:
            return ""
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="ignore")
        return str(data)

    def fetch_chunk(chunk_ids: List[str], depth: int = 0) -> List[SeqRecord]:
        chunk_size = len(chunk_ids)

        for attempt in range(1, EFETCH_RETRIES + 1):
            try:
                with Entrez.efetch(
                    db="nucleotide",
                    id=",".join(chunk_ids),
                    rettype="fasta",
                    retmode="text",
                ) as handle:
                    try:
                        raw_text = to_text(handle.read())
                    except http.client.IncompleteRead as e:
                        partial_text = to_text(getattr(e, "partial", b""))
                        if not partial_text:
                            raise
                        print(
                            f"[WARN] efetch incomplete read (size={chunk_size}, attempt={attempt}/{EFETCH_RETRIES}) "
                            f"-> using partial payload"
                        )
                        raw_text = partial_text

                parsed_chunk = parse_fasta_payload(raw_text)
                if parsed_chunk:
                    return parsed_chunk

                print(
                    f"[WARN] No valid FASTA parsed (size={chunk_size}, attempt={attempt}/{EFETCH_RETRIES})"
                )
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                http.client.IncompleteRead,
                TimeoutError,
                socket.timeout,
                ConnectionError,
                OSError,
            ) as e:
                print(
                    f"[WARN] efetch chunk failed (size={chunk_size}, attempt={attempt}/{EFETCH_RETRIES}): {e}"
                )

            if attempt < EFETCH_RETRIES:
                sleep_s = EFETCH_BACKOFF_SEC * attempt
                print(f"[WARN] retry efetch size={chunk_size} after {sleep_s:.1f}s")
                time.sleep(sleep_s)

        if chunk_size > 1:
            mid = chunk_size // 2
            left = chunk_ids[:mid]
            right = chunk_ids[mid:]
            print(f"[WARN] split efetch chunk {chunk_size} -> {len(left)} + {len(right)}")
            return fetch_chunk(left, depth + 1) + fetch_chunk(right, depth + 1)

        print(f"[WARN] skip efetch id={chunk_ids[0]} after retries")
        return []

    for chunk in batched(ids, EFETCH_BATCH_SIZE):
        records.extend(fetch_chunk(chunk))
    return records


def choose_unique_strain_samples(
    records: List[SeqRecord],
    genus: str,
    species: str,
    profile: Dict[str, object],
    target_count: int,
    pre_used_accessions: Optional[set] = None,
    pre_used_strains: Optional[set] = None,
) -> List[Tuple[SeqRecord, str, str]]:
    selected: List[Tuple[SeqRecord, str, str]] = []
    used_strains = set(pre_used_strains or set())
    used_accessions = set(pre_used_accessions or set())

    min_len = int(profile["min_len"])
    max_len = int(profile["max_len"])
    require_species_phrase_default = bool(profile.get("require_species_phrase", True))
    organism_phrase = build_organism_name(genus, species, bool(profile["join_genus"])).lower()

    def pick(require_species_phrase: bool) -> None:
        for rec in records:
            if len(selected) >= target_count:
                return

            header = rec.description
            seq_len = len(rec.seq)

            if seq_len < min_len or seq_len > max_len:
                continue
            if require_species_phrase and organism_phrase not in header.lower():
                continue
            if is_non_target(header, profile):
                continue

            accession = normalize_token(extract_accession(header))
            strain = extract_strain_key(header)

            if accession in used_accessions:
                continue
            if strain in used_strains:
                continue

            used_accessions.add(accession)
            used_strains.add(strain)
            selected.append((rec, accession, strain))

    # Pass 1: strict
    pick(require_species_phrase_default)

    # Pass 2: relaxed phrase check (hữu ích cho species có synonym/taxonomy rename)
    if len(selected) < target_count and require_species_phrase_default:
        pick(False)

    return selected


def species_prefix(genus: str, species: str) -> str:
    g = next((c for c in genus if c.isalnum()), "X")
    s = next((c for c in species if c.isalnum()), "X")
    return f"{g}{s}".upper()


def species_output_dir(root: Path, genus: str, species: str) -> Path:
    return root / safe_path_name(genus) / safe_path_name(species)


def extract_sample_index(path: Path, prefix: str) -> int:
    m = re.search(rf"^{re.escape(prefix)}_sample(\d+)\.fasta$", path.name, flags=re.IGNORECASE)
    if not m:
        return -1
    return int(m.group(1))


def parse_local_fasta(path: Path) -> Optional[SeqRecord]:
    for fmt in (FASTA_PARSE_FORMAT, "fasta-pearson", "fasta"):
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                rec = next(SeqIO.parse(handle, fmt), None)
                if rec is not None:
                    return rec
        except (ValueError, OSError):
            continue
    return None


def existing_species_state(root: Path, genus: str, species: str) -> Dict[str, object]:
    out_dir = species_output_dir(root, genus, species)
    prefix = species_prefix(genus, species)

    if not out_dir.exists():
        return {
            "count": 0,
            "valid_files": [],
            "accessions": set(),
            "strains": set(),
            "next_index": 1,
        }

    sample_files = []
    for p in out_dir.glob(f"{prefix}_sample*.fasta"):
        idx = extract_sample_index(p, prefix)
        if idx > 0:
            sample_files.append((idx, p))
    sample_files.sort(key=lambda x: x[0])

    valid_files = []
    used_accessions = set()
    used_strains = set()
    max_idx = 0

    for idx, p in sample_files:
        max_idx = max(max_idx, idx)
        rec = parse_local_fasta(p)
        if rec is None:
            continue
        header = rec.description
        accession = normalize_token(extract_accession(header))
        strain = extract_strain_key(header)
        used_accessions.add(accession)
        used_strains.add(strain)
        valid_files.append(p)

    return {
        "count": len(valid_files),
        "valid_files": valid_files,
        "accessions": used_accessions,
        "strains": used_strains,
        "next_index": max_idx + 1 if max_idx > 0 else 1,
    }


def write_species_samples(
    root: Path,
    genus: str,
    species: str,
    selected: List[Tuple[SeqRecord, str, str]],
    append: bool = False,
    start_index: int = 1,
) -> Path:
    out_dir = species_output_dir(root, genus, species)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = species_prefix(genus, species)

    if not append:
        for old_file in out_dir.glob(f"{prefix}_sample*.fasta"):
            old_file.unlink()

    idx = max(1, start_index)
    for rec, _, _ in selected:
        while (out_dir / f"{prefix}_sample{idx}.fasta").exists():
            idx += 1
        file_name = f"{prefix}_sample{idx}.fasta"
        SeqIO.write(rec, out_dir / file_name, "fasta")
        idx += 1

    return out_dir


def count_existing_samples(root: Path, genus: str, species: str) -> int:
    state = existing_species_state(root, genus, species)
    return int(state["count"])


def zip_outputs(roots: List[Path], zip_name: str = "pathogen_dataset.zip") -> Path:
    zip_path = Path(zip_name)
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root in roots:
            if not root.exists():
                continue
            for p in root.rglob("*"):
                if p.is_dir() or p == zip_path:
                    continue
                zf.write(p, arcname=str(p))
    return zip_path


def send_email_report(subject: str, body: str, attachment: Optional[Path] = None) -> bool:
    required = [SMTP_HOST, SMTP_USER, SMTP_PASSWORD, EMAIL_TO]
    if not all(required):
        print("[Email] Skip: thiếu SMTP_HOST/SMTP_USER/SMTP_PASSWORD/EMAIL_TO")
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = EMAIL_TO
    msg.set_content(body)

    if attachment and attachment.exists():
        data = attachment.read_bytes()
        msg.add_attachment(
            data,
            maintype="application",
            subtype="zip",
            filename=attachment.name,
        )

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)

    print(f"[Email] Sent -> {EMAIL_TO}")
    return True


def run_audit_collection(collection: Dict[str, object], tsv_rows: List[Dict[str, object]]) -> str:
    label = str(collection["label"])
    root = Path(collection["root"])
    profile_name = str(collection["profile"])
    profile = SEARCH_PROFILES[profile_name]
    target_count = target_samples_for_profile(profile_name)
    genus_to_species = collection["genus_to_species"]

    root.mkdir(parents=True, exist_ok=True)
    report_lines = [f"Dataset: {label}"]

    for genus, species_list in genus_to_species.items():
        report_lines.append(f"Genus: {genus}")

        for species in species_list:
            local_count = count_existing_samples(root, genus, species)
            missing = max(0, target_count - local_count)
            status = "FULL" if missing == 0 else "MISSING"
            ncbi_ids = ""

            if AUDIT_CHECK_NCBI and missing > 0:
                ids = search_ids(genus, species, profile, target_count=target_count)
                ncbi_ids = str(len(ids))

            line = f"- {species}: local={local_count}/{target_count} missing={missing}"
            if ncbi_ids:
                line += f" ncbi_ids={ncbi_ids}"
            line += f" [{status}]"
            report_lines.append(line)

            tsv_rows.append(
                {
                    "dataset": label,
                    "genus": genus,
                    "species": species,
                    "local_count": local_count,
                    "target_count": target_count,
                    "missing_count": missing,
                    "status": status,
                    "ncbi_ids": ncbi_ids,
                }
            )

    return "\n".join(report_lines)


def run_audit_all_collections() -> str:
    blocks = []
    rows: List[Dict[str, object]] = []

    for collection in DATASET_COLLECTIONS:
        profile_name = str(collection["profile"]).lower()
        if profile_name not in DATASET_TARGETS:
            continue
        blocks.append(run_audit_collection(collection, rows))

    AUDIT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_REPORT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "genus",
                "species",
                "local_count",
                "target_count",
                "missing_count",
                "status",
                "ncbi_ids",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[AUDIT] TSV saved: {AUDIT_REPORT_PATH.resolve()}")
    return "\n\n".join(blocks)


def run_collection(collection: Dict[str, object]) -> str:
    label = str(collection["label"])
    root = Path(collection["root"])
    profile_name = str(collection["profile"])
    profile = SEARCH_PROFILES[profile_name]
    target_count = target_samples_for_profile(profile_name)
    genus_to_species = collection["genus_to_species"]

    root.mkdir(parents=True, exist_ok=True)
    report_lines = [f"Dataset: {label}"]

    for genus, species_list in genus_to_species.items():
        report_lines.append(f"Genus: {genus}")

        for species in species_list:
            display_name = build_organism_name(genus, species, bool(profile["join_genus"]))
            existing = existing_species_state(root, genus, species)
            existing_count = int(existing["count"])

            if SKIP_COMPLETED and not FORCE_REBUILD and existing_count >= target_count:
                report_lines.append(
                    f"- {species}: {target_count}/{target_count} [SKIPPED EXISTING]"
                )
                print(f"[{label}] SKIP {genus}/{species}: already has {existing_count} samples")
                continue

            append_mode = REFILL_ONLY_MISSING and not FORCE_REBUILD
            needed = max(0, target_count - existing_count) if append_mode else target_count

            if needed == 0:
                report_lines.append(
                    f"- {species}: {target_count}/{target_count} [NO REFILL NEEDED]"
                )
                continue

            print(
                f"\n[{label}] Searching: {display_name} "
                f"(existing={existing_count}, need={needed}, append={append_mode})"
            )
            ids = search_ids(genus, species, profile, target_count=target_count)
            print(f"[{label}] IDs found: {len(ids)}")

            if not ids:
                final_count = existing_count if append_mode else 0
                warn = " [NOT ENOUGH UNIQUE STRAINS]" if final_count < target_count else ""
                report_lines.append(f"- {species}: {final_count}/{target_count} (no records){warn}")
                continue

            records = fetch_fasta_records(ids)
            selected = choose_unique_strain_samples(
                records=records,
                genus=genus,
                species=species,
                profile=profile,
                target_count=needed,
                pre_used_accessions=set(existing["accessions"]) if append_mode else None,
                pre_used_strains=set(existing["strains"]) if append_mode else None,
            )

            write_species_samples(
                root=root,
                genus=genus,
                species=species,
                selected=selected,
                append=append_mode,
                start_index=int(existing["next_index"]) if append_mode else 1,
            )

            final_count = existing_count + len(selected) if append_mode else len(selected)

            notes = []
            if append_mode and existing_count > 0:
                notes.append(f"+{len(selected)} new")
            if len(selected) == 0 and len(ids) > 0:
                notes.append("QUERY_HIT_BUT_FILTERED_OR_DUPLICATE")
            if final_count < target_count:
                notes.append("NOT ENOUGH UNIQUE STRAINS")

            suffix = f" [{' | '.join(notes)}]" if notes else ""
            status = f"{final_count}/{target_count}"
            report_lines.append(f"- {species}: {status}{suffix}")
            print(f"[{label}] SAVE {genus}/{species}: {status}{suffix}")

    return "\n".join(report_lines)


def run_all_collections() -> str:
    blocks = []
    for collection in DATASET_COLLECTIONS:
        profile_name = str(collection["profile"]).lower()
        if profile_name not in DATASET_TARGETS:
            continue
        blocks.append(run_collection(collection))
    return "\n\n".join(blocks) if blocks else "No dataset target selected."


def main() -> None:
    configure_ssl_for_entrez()
    print(
        "[CONFIG] targets="
        + ",".join(sorted(DATASET_TARGETS))
        + f" skip_completed={SKIP_COMPLETED}"
        + f" force_rebuild={FORCE_REBUILD}"
        + f" refill_only_missing={REFILL_ONLY_MISSING}"
        + f" audit_only={AUDIT_ONLY}"
        + f" dna_samples_per_species={DNA_SAMPLES_PER_SPECIES}"
        + f" rna_samples_per_species={RNA_SAMPLES_PER_SPECIES}"
        + f" rna_require_complete_genome={int(RNA_REQUIRE_COMPLETE_GENOME)}"
    )
    try:
        if AUDIT_ONLY:
            report = run_audit_all_collections()
        else:
            report = run_all_collections()
        print("\n===== SUMMARY =====")
        print(report)

        attachment = None
        if EMAIL_ATTACH_ZIP:
            roots = [Path(cfg["root"]) for cfg in DATASET_COLLECTIONS]
            attachment = zip_outputs(roots)

        send_email_report(
            subject="NCBI Pathogen Collection Report",
            body=report,
            attachment=attachment,
        )
    except urllib.error.URLError as e:
        msg = str(e)

        if "CERTIFICATE_VERIFY_FAILED" in msg:
            print("\n[ERROR] SSL certificate verify failed khi gọi NCBI.")
            print("Cách xử lý nhanh:")
            print("1) python3 -m pip install certifi")
            print("2) export NCBI_SSL_CA_FILE=\"$(python3 -c 'import certifi; print(certifi.where())')\"")
            print("3) python3 \"import file.py\"")
            print("Tạm thời (không khuyến nghị lâu dài): export NCBI_INSECURE_SSL=1")
            return
        raise


if __name__ == "__main__":
    main()
