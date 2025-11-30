# csv_processing.py
import csv
from typing import List

def extract_text_from_csv(path: str, delimiter: str = ";") -> str:
    """
    Extracts all text from a CSV file and concatenates it into a single string.
    Each row is joined by spaces, and each line ends with a newline.
    """
    rows: List[str] = []

    # Essayer UTF-8 d'abord, puis Latin-1 en fallback
    encodings = ["utf-8", "latin-1", "cp1252"]

    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding, newline="") as f:
                reader = csv.reader(f, delimiter=delimiter)
                for row in reader:
                    if not row:
                        continue
                    cleaned = " ".join(cell.strip() for cell in row if cell.strip())
                    if cleaned:
                        rows.append(cleaned)
            break  # Succès, sortir de la boucle
        except UnicodeDecodeError:
            rows.clear()  # Réinitialiser pour le prochain essai
            continue

    return "\n".join(rows)
