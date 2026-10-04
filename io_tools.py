"""Read tabular files: CSV, delimited text, Excel and JSON."""
import io
import json
import pandas as pd

KINDS = ["Auto-detect", "CSV", "Text / TSV", "Excel", "JSON"]
EXTS = {"Auto-detect": ["csv", "tsv", "txt", "xlsx", "xlsm", "xls", "json"], "CSV": ["csv"],
        "Text / TSV": ["tsv", "txt", "csv"], "Excel": ["xlsx", "xlsm", "xls"], "JSON": ["json"]}
SEPS = {"Auto-detect": None, "Comma": ",", "Semicolon": ";", "Tab": "\t", "Pipe": "|"}


def detect_kind(name):
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    for kind, exts in (("CSV", ["csv"]), ("Text / TSV", ["tsv", "txt"]), ("Excel", ["xlsx", "xlsm", "xls"]), ("JSON", ["json"])):
        if ext in exts:
            return kind
    raise ValueError(f"Cannot tell the file type from '{name}'. Choose the file type manually.")


def guess_sep(data):
    """Pick the delimiter that appears most consistently in the first lines (more reliable than csv.Sniffer on numeric data)."""
    text = data[:20000].decode("utf-8", errors="replace").splitlines()[:8]
    text = [t for t in text if t.strip()]
    best, best_score = None, 0
    for cand in [",", ";", "\t", "|"]:
        counts = [t.count(cand) for t in text]
        if counts and min(counts) > 0 and len(set(counts)) <= 2 and min(counts) > best_score:
            best, best_score = cand, min(counts)
    return best


def excel_sheets(data):
    try:
        return pd.ExcelFile(io.BytesIO(data)).sheet_names
    except Exception as e:
        raise ValueError(f"Could not open this Excel file ({e}). Old .xls files need the 'xlrd' package; try saving as .xlsx.")


def read_table(data, kind, sep_name="Auto-detect", sheet=None):
    try:
        if kind in ("CSV", "Text / TSV"):
            sep = SEPS[sep_name] if SEPS[sep_name] is not None else guess_sep(data); df = None
            for enc in ("utf-8-sig", "latin-1"):
                try:
                    df = pd.read_csv(io.BytesIO(data), sep=sep, engine="python" if sep is None else "c", encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
        elif kind == "Excel":
            df = pd.read_excel(io.BytesIO(data), sheet_name=sheet if sheet is not None else 0)
        elif kind == "JSON":
            try:
                df = pd.read_json(io.BytesIO(data))
            except ValueError:
                df = pd.json_normalize(json.loads(data))
        else:
            raise ValueError(f"Unknown file type {kind}")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Could not read this file as {kind}: {e}")
    df = df.dropna(how="all").dropna(axis=1, how="all")
    names, seen = [], {}
    for c in df.columns:
        c = str(c).strip(); seen[c] = seen.get(c, 0) + 1
        names.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
    df.columns = names
    if df.empty or df.shape[1] == 0:
        raise ValueError("The file has no usable rows or columns.")
    return df.reset_index(drop=True)
