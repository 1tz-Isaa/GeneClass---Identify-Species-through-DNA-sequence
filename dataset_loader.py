import os


def read_fasta(path):
    sequences = []
    header = None
    seq = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    sequences.append((header, "".join(seq)))
                header = line[1:]
                seq = []
            else:
                seq.append(line)
        if header:
            sequences.append((header, "".join(seq)))
    return sequences


def infer_labels(root_folder, dirpath, filename):
    root_name = os.path.basename(os.path.normpath(root_folder))
    rel_parts = os.path.relpath(dirpath, root_folder).split(os.sep)

    domain = root_name
    kingdom = rel_parts[0] if len(rel_parts) >= 1 else "unknown"
    genus = rel_parts[1] if len(rel_parts) >= 2 else "unknown"
    species = rel_parts[2] if len(rel_parts) >= 3 else os.path.splitext(filename)[0]

    return domain, kingdom, genus, species


def load_dataset(root_folder, show_progress=False):
    data = []

    root_folder = os.path.abspath(root_folder)
    fasta_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith(".fasta"):
                fasta_files.append((dirpath, file))

    total_files = len(fasta_files)
    total_records = 0

    for idx, (dirpath, file) in enumerate(fasta_files, start=1):
        full_path = os.path.join(dirpath, file)

        domain, kingdom, genus, species = infer_labels(root_folder, dirpath, file)

        sequences = read_fasta(full_path)
        total_records += len(sequences)

        if show_progress:
            print(
                f"[LOAD] {idx}/{total_files} file={full_path} "
                f"records_in_file={len(sequences)} total_records={total_records}"
            )

        for header, seq in sequences:
            data.append({
                "sequence": seq,
                "domain": domain,
                "kingdom": kingdom,
                "genus": genus,
                "species": species,
                "source_file": full_path,
                "header": header,
            })

    return data
