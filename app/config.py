VALID_SUBJECTS = [
    "balagha",
    "eirab",
    "qeraah",
    "adab_nosoos",
    "nahw",
    "taabir",
    "qessa",
]

SUBJECT_CHUNK_CONFIG = {
    "eirab": {"max_chars": 400, "overlap": 50},
    "nahw": {"max_chars": 500, "overlap": 100},
    "balagha": {"max_chars": 500, "overlap": 100},
    "qeraah": {"max_chars": 700, "overlap": 150},
    "adab_nosoos": {"max_chars": 700, "overlap": 150},
    "taabir": {"max_chars": 600, "overlap": 100},
    "qessa": {"max_chars": 600, "overlap": 100},
}

TOP_K_BY_SUBJECT = {
    "eirab": 2,
    "nahw": 3,
    "balagha": 3,
    "qeraah": 4,
    "adab_nosoos": 4,
    "taabir": 4,
    "qessa": 4,
}