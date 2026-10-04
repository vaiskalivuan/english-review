#!/usr/bin/env python3
"""Build a compact, de-identified data.json from personal markdown study notes.

Usage:
    python3 build_data.py --src <notes dir> --exclude-terms <terms file> \
        [--out data.json] [--report report.md] [--preview preview.md]

The notes directory is scanned for markdown files, which are classified by
their first heading (not by file name):
    - vocabulary list      (heading contains the word list title)
    - error log            (heading contains the error log title)
    - lesson log           (heading contains the lesson log title)
    - memorization list    (heading contains the memorization list title)
    - materials digest     (heading contains the materials title)
    - supplement           (heading contains the supplement title)

Output has three collections: vocab, errors, pronunciation.
Each item has a stable id derived from its content.

Privacy:
    * No teacher names, dates, lesson summaries or discussion topics are read
      into the output.
    * --exclude-terms points to a plain-text file (one term per line, '#'
      comments allowed, case-insensitive substring match). It is kept OUTSIDE
      this repository. A leading '!' marks a strict term (checked against
      words and the final output too); other terms apply to sentences only.
      Any error item that matches is dropped entirely; any example sentence
      that matches is dropped (the vocab item is kept).
    * Example sentences from tailored sources are never read.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# ---- file classification (by first heading) -------------------------------
KIND_BY_HEADING = (
    ("單字庫", "vocab"),
    ("錯誤本", "errors"),
    ("課堂紀錄", "lessons"),
    ("背誦清單", "memo"),
    ("單字與文法", "materials"),
    ("AI補充", "supp"),
)
# Sources whose example sentences were written for one person: never read them.
TAILORED_KINDS = {"memo"}

# ---- error type normalisation ---------------------------------------------
CATEGORIES = (
    ("時態", ("時態", "過去式", "現在完成", "進行式", "動詞形態", "動詞形式", "原形")),
    ("主謂一致", ("主謂一致",)),
    ("單複數", ("單複數", "複數", "所有格")),
    ("冠詞", ("冠詞",)),
    ("介系詞", ("介系詞",)),
    ("詞性", ("詞性", "-ed", "-ing", "名詞", "形容詞")),
    ("用字", ("用字", "用法", "片語", "搭配", "精準", "更自然", "自然度")),
    ("語序", ("語序", "句型", "缺 be", "缺 is", "缺動詞", "問句", "斷句", "斷開", "否定", "結構")),
)
NOTE_REJECT = re.compile(r"老師|總評|\d{1,2}-\d{1,2}|堂")

CJK = re.compile(r"[㐀-鿿]")
LEADING_FUNC = {"a", "an", "the", "my", "our", "your", "some", "this", "that", "was", "is", "are"}
TRAILING_FUNC = {"in", "on", "at", "about", "of", "to", "very", "actually"}

WORD_HEADERS = ("單字", "字", "片語")
MEANING_PREFIXES = ("中文", "解釋", "完整意思", "意思")


def sid(prefix, *parts):
    key = "|".join(re.sub(r"\s+", " ", p.strip().lower()) for p in parts)
    return f"{prefix}_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:8]}"


def load_terms(path):
    """Return (strict, soft). A leading '!' marks a strict term.

    strict: checked against everything (words, meanings, sentences, final output)
    soft:   checked against sentences only (error items and example sentences)
    """
    strict, soft = [], []
    if path:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("!"):
                strict.append(line[1:].strip().lower())
            else:
                soft.append(line.lower())
    return strict, soft


def hit(text, terms):
    low = text.lower()
    for t in terms:
        if t in low:
            return t
    return None


def classify(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            for key, kind in KIND_BY_HEADING:
                if key in line:
                    return kind
            return None
    return None


# ---- markdown helpers ------------------------------------------------------
def split_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def tables(text):
    """Yield (header_cells, [row_cells]) for each markdown table."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            header = split_row(lines[i])
            rows = []
            i += 2
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            yield header, rows
        else:
            i += 1


def clean_meaning(s):
    s = re.sub(r"（[^）]*(老師|我補的)[^）]*）", "", s)
    s = s.replace("（我補的）", "")
    return s.strip()


def clean_example(s):
    s = s.strip().strip('"').strip("“”").strip()
    if not s or s in {"—", "-"}:
        return ""
    if CJK.search(s) or s.startswith("（") or s.startswith("("):
        return ""
    return s


def clean_word(s):
    s = re.sub(r"\s*（[^）]*）\s*", "", s)  # drop full-width notes
    s = re.sub(r"\s*\(.*?\)\s*$", "", s).strip()  # drop trailing "(abbr)" notes
    return s.strip('"').strip()


def split_definition(cell):
    """'<definition>. <example sentence>' -> (definition, example)."""
    cell = cell.strip()
    m = re.search(r"(?<=[a-z\)])\.\s+(?=[A-Z])", cell)
    if not m:
        return cell.rstrip("."), ""
    return cell[: m.start()].strip(), cell[m.end():].strip()


# ---- collector -------------------------------------------------------------
class Collector:
    def __init__(self, strict, soft):
        self.strict = strict
        self.sentence_terms = strict + soft
        self.vocab = {}
        self.errors = {}
        self.pron = {}
        self.ipa_hints = {}
        self.excluded = []  # (kind, text, reason)

    def add_vocab(self, word, meaning="", example="", ipa="", definition="", tailored=False):
        word = clean_word(word)
        meaning = clean_meaning(meaning)
        if not word or not re.search(r"[A-Za-z]", word):
            return
        if tailored:
            example = ""
            definition = ""
        example = clean_example(example)
        t = hit(word + " " + meaning + " " + definition, self.strict)
        if t:
            self.excluded.append(("vocab", word, t))
            return
        t = hit(example, self.sentence_terms) if example else None
        if t:
            self.excluded.append(("example", example, t))
            example = ""
        key = word.lower()
        cur = self.vocab.get(key)
        if cur:
            for k, v in (("meaning", meaning), ("example", example), ("ipa", ipa), ("definition", definition)):
                if v and not cur[k]:
                    cur[k] = v
        else:
            self.vocab[key] = {
                "id": sid("w", word),
                "word": word,
                "ipa": ipa,
                "meaning": meaning,
                "definition": definition,
                "example": example,
            }

    def add_error(self, wrong, correct, type_text="", force_types=None):
        wrong = re.sub(r"^[\.…\s]+", "", wrong).strip()
        correct = re.sub(r"^[\.…\s]+", "", correct).strip()
        if not wrong or not correct or wrong == correct:
            return
        t = hit(wrong + " " + correct, self.sentence_terms)
        if t:
            self.excluded.append(("error", f"{wrong}  =>  {correct}", t))
            return
        types = list(force_types or [])
        for name, keys in CATEGORIES:
            if any(k in type_text for k in keys) and name not in types:
                types.append(name)
        if not types:
            types = ["其他"]
        note = type_text.strip()
        if NOTE_REJECT.search(note) or hit(note, self.sentence_terms):
            note = ""
        key = (wrong.lower(), correct.lower())
        if key not in self.errors:
            self.errors[key] = {
                "id": sid("e", wrong, correct),
                "wrong": wrong,
                "correct": correct,
                "types": types,
                "note": note,
            }

    def add_pron(self, word, ipa="", hint=""):
        t = hit(word + " " + hint, self.strict)
        if t:
            self.excluded.append(("pronunciation", word, t))
            return
        key = word.lower()
        cur = self.pron.get(key)
        if cur:
            if ipa and not cur["ipa"]:
                cur["ipa"] = ipa
            if hint and not cur["hint"]:
                cur["hint"] = hint
        else:
            self.pron[key] = {"id": sid("p", word), "word": word, "ipa": ipa, "hint": hint}

    def hint_ipa(self, word, ipa):
        self.ipa_hints[word.strip().lower()] = ipa


# ---- parsers ---------------------------------------------------------------
def parse_vocab_tables(text, col, tailored):
    for header, rows in tables(text):
        idx = {}
        for n, h in enumerate(header):
            if h in WORD_HEADERS:
                idx["word"] = n
            elif h.startswith(MEANING_PREFIXES):
                idx["meaning"] = n
            elif h.startswith("定義與例句"):
                idx["defex"] = n
            elif h.startswith("例句") or h == "例":
                idx["example"] = n
            elif h == "音標":
                idx["ipa"] = n
        get = lambda r, k: r[idx[k]] if k in idx and idx[k] < len(r) else ""
        if "word" in idx and ("meaning" in idx):
            for r in rows:
                definition, example = "", get(r, "example")
                if "defex" in idx:
                    definition, example = split_definition(get(r, "defex"))
                col.add_vocab(get(r, "word"), get(r, "meaning"), example, get(r, "ipa"), definition, tailored)
        elif "word" in idx and "ipa" in idx:
            for r in rows:
                if get(r, "word") and get(r, "ipa"):
                    col.hint_ipa(get(r, "word"), get(r, "ipa"))
        elif "簡單說法" in header and "專業說法" in header:
            a, b = header.index("簡單說法"), header.index("專業說法")
            for r in rows:
                # long entries are whole sentences, not study words
                if len(r) > max(a, b) and len(r[b].split()) <= 5:
                    col.add_vocab(r[b], f"「{r[a]}」的專業說法", "", "", "", tailored)
        elif "你說的" in header and any(h.startswith("更自然") for h in header):
            a = header.index("你說的")
            b = next(n for n, h in enumerate(header) if h.startswith("更自然"))
            for r in rows:
                if len(r) > max(a, b):
                    col.add_error(r[a], r[b], "用字（更自然說法）", force_types=["用字"])
    for line in text.splitlines():
        # bullet pairs: "- roll out = 推出、上線（introduce）"
        m = re.match(r"^\s*-\s*([A-Za-z][A-Za-z' \-]*?)\s*=\s*(.+)$", line)
        if m:
            col.add_vocab(m.group(1), m.group(2), "", "", "", tailored)
            continue
        # extension words: "**延伸詞**：minister（部長）、investment（投資）"
        m = re.match(r"^\*\*延伸詞\*\*：(.*)$", line.strip())
        if m:
            for tok in m.group(1).split("、"):
                mm = re.match(r"^\s*([A-Za-z][A-Za-z' \-\.]*?)\s*（([^）]+)）\s*$", tok)
                if mm and CJK.search(mm.group(2)):
                    col.add_vocab(mm.group(1), mm.group(2), "", "", "", tailored)


def parse_errors(text, col):
    text = text.split("## 重複出現的問題")[0]
    cur = None

    def flush():
        nonlocal cur
        if cur and cur.get("wrong") and cur.get("correct"):
            col.add_error(cur["wrong"], cur["correct"], cur.get("type", ""))
        cur = None

    for line in text.splitlines():
        m = re.match(r"^\s*\d+\.\s*❌\s*(.*)$", line)
        if m:
            flush()
            cur = {"wrong": m.group(1).strip()}
            continue
        if cur is None:
            continue
        m = re.match(r"^\s*✔️?\s*(.*)$", line)
        if m and "correct" not in cur:
            cur["correct"] = m.group(1).strip()
            continue
        m = re.match(r"^\s*類型：(.*)$", line)
        if m and "type" not in cur:
            cur["type"] = m.group(1).strip()
            continue
        if line.startswith("## ") or line.strip() == "---":
            flush()
    flush()


def parse_pron_token(tok):
    """Return (word, ipa, hint) or None."""
    tok = tok.strip().strip("。.")
    if not tok:
        return None
    ipa = ""
    m = re.search(r"/([^/]+)/", tok)
    if m:
        ipa = "/" + m.group(1) + "/"
        tok = (tok[: m.start()] + tok[m.end():]).strip()
    tok = re.sub(r"（[^）]*）", "", tok).strip()
    hint = ""
    if "→" in tok:
        tok, hint = [p.strip() for p in tok.split("→", 1)]
    words = [w for w in re.split(r"\s+", tok) if w]
    while words and words[0].lower() in LEADING_FUNC:
        words = words[1:]
    while words and words[-1].lower().strip(",.") in TRAILING_FUNC:
        words = words[:-1]
    if not words or len(words) > 2:
        return None
    word = " ".join(words)
    if not re.fullmatch(r"[A-Za-z][A-Za-z' \-]*", word):
        return None
    return word, ipa, hint


def parse_pron_lines(text, col):
    for line in text.splitlines():
        m = re.match(r"^\*\*發音(?:字|（\d+ 則）)\*\*：(.*)$", line.strip())
        if not m:
            continue
        # ' / ' separates phrase lists; IPA slashes have no surrounding spaces
        for part in re.split(r"\s/\s", m.group(1)):
            for tok in re.split(r"、|,\s*", part):
                got = parse_pron_token(tok)
                if got:
                    col.add_pron(*got)


# ---- main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory with the markdown notes")
    ap.add_argument("--exclude-terms", help="text file of terms that must not be published")
    ap.add_argument("--out", default="data.json")
    ap.add_argument("--report", help="write the exclusion report here (keep outside the repo)")
    ap.add_argument("--preview", help="write a full human-readable preview here (keep outside the repo)")
    args = ap.parse_args()

    strict, soft = load_terms(args.exclude_terms)
    if not (strict or soft):
        print("WARNING: no exclude terms loaded; privacy filter is OFF", file=sys.stderr)
    col = Collector(strict, soft)

    kinds = {}
    for p in sorted(Path(args.src).glob("*.md")):
        k = classify(p)
        if k:
            kinds[p] = k
    missing = {"vocab", "errors", "lessons", "memo"} - set(kinds.values())
    if missing:
        print(f"WARNING: no source found for: {sorted(missing)}", file=sys.stderr)

    for p, k in kinds.items():
        text = p.read_text(encoding="utf-8")
        if k in ("vocab", "memo", "materials", "supp"):
            parse_vocab_tables(text, col, tailored=k in TAILORED_KINDS)
        elif k == "errors":
            parse_errors(text, col)
        elif k == "lessons":
            parse_pron_lines(text, col)

    # share IPA between vocab and pronunciation, then apply supplement hints
    for key, v in col.vocab.items():
        pr = col.pron.get(key)
        if pr and pr["ipa"] and not v["ipa"]:
            v["ipa"] = pr["ipa"]
    for key, pr in col.pron.items():
        v = col.vocab.get(key)
        if v and v["ipa"] and not pr["ipa"]:
            pr["ipa"] = v["ipa"]
    for key, ipa in col.ipa_hints.items():
        if key in col.pron and not col.pron[key]["ipa"]:
            col.pron[key]["ipa"] = ipa
        if key in col.vocab and not col.vocab[key]["ipa"]:
            col.vocab[key]["ipa"] = ipa

    vocab = sorted(col.vocab.values(), key=lambda x: x["word"].lower())
    errors = list(col.errors.values())
    pron = sorted(col.pron.values(), key=lambda x: x["word"].lower())

    data = {"schema": 1, "vocab": vocab, "errors": errors, "pronunciation": pron}
    out = json.dumps(data, ensure_ascii=False, indent=1) + "\n"

    # last line of defence: published text must not contain any strict term
    leak = hit(out, strict)
    if leak:
        print(f"ABORT: output still contains an excluded term: {leak!r}", file=sys.stderr)
        sys.exit(2)

    Path(args.out).write_text(out, encoding="utf-8")

    summary = (
        f"vocab: {len(vocab)} (meaning: {sum(1 for v in vocab if v['meaning'])}, "
        f"example: {sum(1 for v in vocab if v['example'])}, ipa: {sum(1 for v in vocab if v['ipa'])})\n"
        f"errors: {len(errors)}\n"
        f"pronunciation: {len(pron)} (ipa: {sum(1 for p in pron if p['ipa'])})\n"
        f"excluded: {len(col.excluded)}"
    )
    print(summary)

    if args.report:
        lines = ["# Excluded items", ""]
        for kind, text, why in col.excluded:
            lines.append(f"- [{kind}] (matched {why!r}) {text}")
        Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.preview:
        lines = ["# Preview of data.json", "", summary.replace("\n", "  \n"), "", "## vocab", ""]
        for v in vocab:
            lines.append(f"- **{v['word']}** {v['ipa']} | {v['meaning']} | {v['definition']} | {v['example']}")
        lines += ["", "## errors", ""]
        for e in errors:
            lines.append(f"- ❌ {e['wrong']}  \n  ✔️ {e['correct']}  \n  [{', '.join(e['types'])}] {e['note']}")
        lines += ["", "## pronunciation", ""]
        for p in pron:
            lines.append(f"- {p['word']} {p['ipa']} {p['hint']}")
        Path(args.preview).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
