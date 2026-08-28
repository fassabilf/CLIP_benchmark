#!/usr/bin/env python3
"""Tokenizer audit: student CLIP-BPE vs teacher XLM-V on the evaluated SEA languages.

Produces the evidence asked for by the ACCV'26 #346 reviewers (tokenizer claim in
Sec. 3.3): script coverage of each vocabulary, token lengths, truncation rates,
fertility, and byte-fallback / <unk> behaviour.

Part A -- vocabulary script coverage
    For each tokenizer, how many vocabulary entries contain characters of each
    SEA-relevant script (Thai, Myanmar, Khmer, Lao, Javanese, Sundanese, Balinese,
    Cham, Tai Tham, Tagalog, Batak, Buginese) plus Vietnamese-diacritic Latin.
    For the student we also report how many entries are bare UTF-8 byte fragments,
    which is how a byte-level BPE represents scripts it has no whole-character
    token for (lossless, but at a sequence-length cost).

Part B -- per-language caption statistics at ctx 77
    tokens/caption (mean/median/p95/max), truncation rate, fertility (tokens per
    character and per whitespace word), byte-fragment token share (student) and
    <unk> share (teacher), for XM3600 / Flickr30k-200 / XTD-200 / Babel-ImageNet.

Self-contained: re-implements open_clip's SimpleTokenizer.encode() so no torch or
open_clip import is required. Run --verify-openclip to assert the reimplementation
is token-identical to the real tokenizer used in training.

Typical use on LANTA (transfer node, mc2_eval_env):

    source runs/env.sh
    module load Mamba/23.11.0-0 && source activate mc2_eval_env
    python runs/tokenizer_audit.py --eval-root "$EVAL_ROOT"
"""
import argparse
import gzip
import html
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict

try:
    import regex as re
    _HAS_REGEX = True
except ImportError:
    import re
    _HAS_REGEX = False
    print("[warn] `regex` not installed; falling back to `re` (pattern approximated)", file=sys.stderr)

try:
    import ftfy

    def basic_clean(t):
        return html.unescape(html.unescape(ftfy.fix_text(t))).strip()
except ImportError:
    print("[warn] `ftfy` not installed; skipping fix_text", file=sys.stderr)

    def basic_clean(t):
        return html.unescape(html.unescape(t)).strip()


# --------------------------------------------------------------------- scripts
# Ranges cover the scripts of the seven evaluated SEA languages plus the other SEA
# scripts present in the training blend (Bloom: km, lo, ...).
SCRIPT_RANGES = {
    "Thai":      [(0x0E00, 0x0E7F)],
    "Lao":       [(0x0E80, 0x0EFF)],
    "Myanmar":   [(0x1000, 0x109F), (0xA9E0, 0xA9FF), (0xAA60, 0xAA7F)],
    "Khmer":     [(0x1780, 0x17FF), (0x19E0, 0x19FF)],
    "Tai_Tham":  [(0x1A20, 0x1AAF)],
    "Balinese":  [(0x1B00, 0x1B7F)],
    "Sundanese": [(0x1B80, 0x1BBF), (0x1CC0, 0x1CCF)],
    "Batak":     [(0x1BC0, 0x1BFF)],
    "Javanese":  [(0xA980, 0xA9DF)],
    "Cham":      [(0xAA00, 0xAA5F)],
    "Tagalog":   [(0x1700, 0x171F)],
    "Buginese":  [(0x1A00, 0x1A1F)],
}
# Latin letters carrying Vietnamese diacritics -- vi is Latin-script, so plain
# script ranges would say nothing about how well it is covered.
VI_DIACRITIC_CHARS = set(
    "ăâđêôơưĂÂĐÊÔƠƯ"
    "áàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    "ÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ"
)


def char_script(ch):
    cp = ord(ch)
    for name, ranges in SCRIPT_RANGES.items():
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return name
    if ch in VI_DIACRITIC_CHARS:
        return "Latin_VI_diacritic"
    return None


def classify_string(s):
    return {sc for sc in (char_script(c) for c in s) if sc}


# ---------------------------------------------------------- student tokenizer
def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))


def get_pairs(word):
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


class StudentBPE:
    """Faithful copy of open_clip.tokenizer.SimpleTokenizer (clean='lower')."""

    def __init__(self, bpe_path, context_length=77):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(m.split()) for m in merges]
        vocab = list(self.byte_encoder.values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for m in merges:
            vocab.append("".join(m))
        special_tokens = ["<start_of_text>", "<end_of_text>"]
        vocab.extend(special_tokens)
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {t: t for t in special_tokens}
        self.special_tokens = special_tokens
        self.vocab_size = len(self.encoder)
        self.context_length = context_length
        special = "|".join(special_tokens)
        if _HAS_REGEX:
            pat = special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""
        else:
            pat = special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|[^\s\w]+"""
        self.pat = re.compile(pat, re.IGNORECASE)
        self._text_cache = {}

    def clean(self, text):
        return " ".join(basic_clean(text).split()).strip().lower()

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word, i = [], 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        out = []
        for token in re.findall(self.pat, self.clean(text)):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            out.extend(self.encoder[t] for t in self.bpe(token).split(" "))
        return out

    def token_text(self, token_id):
        """Decoded form of a vocab entry, or None when the entry is a bare UTF-8
        byte fragment that is not a valid character on its own."""
        if token_id in self._text_cache:
            return self._text_cache[token_id]
        s = self.decoder[token_id]
        if s in self.special_tokens:
            out = s
        else:
            try:
                out = bytearray(self.byte_decoder[c] for c in s.replace("</w>", "")).decode("utf-8")
            except (UnicodeDecodeError, KeyError):
                out = None
        self._text_cache[token_id] = out
        return out


# ------------------------------------------------------------- vocab auditing
def audit_student_vocab(stu):
    per_script = Counter()
    frag = 0
    for tid in range(stu.vocab_size):
        txt = stu.token_text(tid)
        if txt is None:
            frag += 1
            continue
        for sc in classify_string(txt):
            per_script[sc] += 1
    return {
        "name": "student CLIP-BPE (byte-level)",
        "vocab_size": stu.vocab_size,
        "byte_fragment_entries": frag,
        "per_script": dict(sorted(per_script.items())),
        "sea_script_units_total": sum(v for k, v in per_script.items() if k != "Latin_VI_diacritic"),
        "vi_diacritic_units": per_script.get("Latin_VI_diacritic", 0),
    }


def audit_teacher_vocab(tok, name):
    per_script = Counter()
    vocab = tok.get_vocab()
    for piece in vocab:
        s = piece.replace("▁", "")
        if not s:
            continue
        for sc in classify_string(s):
            per_script[sc] += 1
    return {
        "name": f"teacher {name}",
        "vocab_size": len(vocab),
        "byte_fragment_entries": None,
        "per_script": dict(sorted(per_script.items())),
        "sea_script_units_total": sum(v for k, v in per_script.items() if k != "Latin_VI_diacritic"),
        "vi_diacritic_units": per_script.get("Latin_VI_diacritic", 0),
    }


# ------------------------------------------------------------- caption loading
XM3600_LANGS = {"en": "en", "id": "id", "th": "th", "vi": "vi"}
FLORES_LANGS = {
    "en": "eng_Latn", "id": "ind_Latn", "jv": "jav_Latn", "ms": "zsm_Latn",
    "my": "mya_Mymr", "su": "sun_Latn", "th": "tha_Thai", "vi": "vie_Latn",
}
BABEL_LANGS = ["en", "id", "jv", "ms", "my", "su", "th", "vi"]


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _flatten(annotations):
    out = []
    for a in annotations:
        if isinstance(a, list):
            out.extend(str(x) for x in a)
        else:
            out.append(str(a))
    return out


def collect_captions(eval_root, babel_path):
    """{(benchmark, lang): [caption, ...]} from whatever is present on disk."""
    out = defaultdict(list)
    if eval_root and os.path.isdir(eval_root):
        for lang, code in XM3600_LANGS.items():
            p = os.path.join(eval_root, f"crossmodal3600_captions-{code}.json")
            if os.path.isfile(p):
                out[("XM3600", lang)] = _flatten(_load_json(p)["annotations"])
        for bench, tmpl in (("Flickr30k-200", "flickr30k_200-{}.json"),
                            ("XTD-200", "xtd200-{}.json")):
            for lang, code in FLORES_LANGS.items():
                p = os.path.join(eval_root, tmpl.format(code))
                if os.path.isfile(p):
                    out[(bench, lang)] = _flatten(_load_json(p)["annotations"])
    if babel_path and os.path.isfile(babel_path):
        d = _load_json(babel_path)
        for lang in BABEL_LANGS:
            entry = d.get(lang.upper(), d.get(lang))
            if entry is None:
                continue
            names = entry[1] if isinstance(entry, list) and len(entry) == 2 else entry
            if isinstance(names, dict):
                names = list(names.values())
            out[("Babel-IN", lang)] = [str(x) for x in names]
    return out


# --------------------------------------------------------------- corpus stats
def pct(x, n):
    return round(100.0 * x / n, 2) if n else 0.0


def p95(sorted_vals):
    return sorted_vals[max(0, min(len(sorted_vals) - 1, int(0.95 * len(sorted_vals)) - 1))]


def corpus_stats(captions, stu, tea, ctx=77):
    rows = []
    for (bench, lang), caps in sorted(captions.items()):
        if not caps:
            continue
        n = len(caps)
        n_chars = sum(len(c) for c in caps)
        n_words = sum(len(c.split()) for c in caps)
        row = {"benchmark": bench, "lang": lang, "n_captions": n,
               "chars_per_caption": round(n_chars / n, 1)}

        s_len, s_frag, s_total = [], 0, 0
        for c in caps:
            ids = stu.encode(c)
            s_len.append(len(ids) + 2)  # <sot> ... <eot>
            s_total += len(ids)
            s_frag += sum(1 for i in ids if stu.token_text(i) is None)
        s_sorted = sorted(s_len)
        row.update({
            "stu_mean": round(statistics.mean(s_len), 1),
            "stu_median": int(statistics.median(s_len)),
            "stu_p95": int(p95(s_sorted)),
            "stu_max": max(s_len),
            "stu_trunc_pct": pct(sum(1 for L in s_len if L > ctx), n),
            "stu_tok_per_char": round(s_total / n_chars, 3) if n_chars else None,
            "stu_tok_per_word": round(s_total / n_words, 2) if n_words else None,
            "stu_bytefrag_pct": pct(s_frag, s_total),
        })

        if tea is not None:
            unk_id = tea.unk_token_id
            t_len, t_unk, t_total = [], 0, 0
            B = 512
            for i in range(0, n, B):
                for ids in tea(caps[i:i + B], add_special_tokens=True)["input_ids"]:
                    t_len.append(len(ids))
                    t_total += len(ids)
                    if unk_id is not None:
                        t_unk += sum(1 for x in ids if x == unk_id)
            t_sorted = sorted(t_len)
            row.update({
                "tea_mean": round(statistics.mean(t_len), 1),
                "tea_median": int(statistics.median(t_len)),
                "tea_p95": int(p95(t_sorted)),
                "tea_max": max(t_len),
                "tea_trunc_pct": pct(sum(1 for L in t_len if L > ctx), n),
                "tea_tok_per_char": round(t_total / n_chars, 3) if n_chars else None,
                "tea_tok_per_word": round(t_total / n_words, 2) if n_words else None,
                "tea_unk_pct": pct(t_unk, t_total),
            })
        rows.append(row)
    return rows


def verify_against_open_clip(stu, captions, n_sample=200, seed=0):
    """Assert the reimplementation is token-identical to open_clip's tokenizer."""
    try:
        from open_clip import get_tokenizer
    except ImportError as e:
        return {"status": "skipped", "reason": f"open_clip not importable ({e})"}
    ref = get_tokenizer("ViT-B-16")
    pool = [c for caps in captions.values() for c in caps]
    if not pool:
        return {"status": "skipped", "reason": "no captions loaded"}
    rng = random.Random(seed)
    sample = rng.sample(pool, min(n_sample, len(pool)))
    mismatch = 0
    for c in sample:
        mine = stu.encode(c)
        theirs = ref.encode(c) if hasattr(ref, "encode") else None
        if theirs is None:
            return {"status": "skipped", "reason": "tokenizer has no .encode()"}
        if mine != theirs:
            mismatch += 1
    return {"status": "ok" if mismatch == 0 else "MISMATCH",
            "n_sampled": len(sample), "n_mismatch": mismatch}


# -------------------------------------------------------------------- output
def to_markdown(audits, rows, ctx):
    L = ["# Tokenizer audit -- student CLIP-BPE vs teacher XLM-V", "",
         "## A. Vocabulary script coverage", ""]
    scripts = sorted({s for a in audits for s in a["per_script"]})
    L.append("| tokenizer | vocab | " + " | ".join(scripts) + " | SEA-script units | byte-fragment entries |")
    L.append("|" + "---|" * (len(scripts) + 4))
    for a in audits:
        cells = [str(a["per_script"].get(s, 0)) for s in scripts]
        L.append(f"| {a['name']} | {a['vocab_size']} | " + " | ".join(cells)
                 + f" | **{a['sea_script_units_total']}** | "
                 + (str(a["byte_fragment_entries"]) if a["byte_fragment_entries"] is not None else "n/a")
                 + " |")
    if rows:
        L += ["", f"## B. Caption statistics (ctx = {ctx}; `stu` = student, `tea` = teacher)", ""]
        keys = [k for k in rows[0] if k not in ("benchmark", "lang")]
        L.append("| benchmark | lang | " + " | ".join(keys) + " |")
        L.append("|" + "---|" * (len(keys) + 2))
        for r in rows:
            L.append(f"| {r['benchmark']} | {r['lang']} | "
                     + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    return "\n".join(L)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpe", default="", help="bpe_simple_vocab_16e6.txt.gz (auto-detected)")
    ap.add_argument("--teacher", default="facebook/xlm-v-base", help="'' to skip the teacher side")
    ap.add_argument("--eval-root", default=os.environ.get("EVAL_ROOT", ""))
    ap.add_argument("--babel", default=os.path.join(repo, "clip_benchmark", "datasets", "babel_imagenet.json"))
    ap.add_argument("--ctx", type=int, default=77)
    ap.add_argument("--vocab-only", action="store_true")
    ap.add_argument("--verify-openclip", action="store_true",
                    help="check StudentBPE.encode() == open_clip tokenizer on a caption sample")
    ap.add_argument("--out-json", default=os.path.join(here, "results", "tokenizer_audit.json"))
    ap.add_argument("--out-md", default=os.path.join(here, "results", "tokenizer_audit.md"))
    args = ap.parse_args()

    bpe = args.bpe
    if not bpe:
        for cand in (
            "/project/lt200394-thllmV/multilingual-clip-kd/open_clip/src/open_clip/bpe_simple_vocab_16e6.txt.gz",
            "/project/lt200394-thllmV/mkd-exp/open_clip/src/open_clip/bpe_simple_vocab_16e6.txt.gz",
            os.path.join(os.path.dirname(repo), "open_clip", "src", "open_clip", "bpe_simple_vocab_16e6.txt.gz"),
        ):
            if os.path.isfile(cand):
                bpe = cand
                break
    if not bpe or not os.path.isfile(bpe):
        sys.exit("could not locate bpe_simple_vocab_16e6.txt.gz; pass --bpe")
    print(f"[info] student BPE: {bpe}", file=sys.stderr)
    stu = StudentBPE(bpe, context_length=args.ctx)

    tea = None
    if args.teacher:
        try:
            from transformers import AutoTokenizer
            tea = AutoTokenizer.from_pretrained(args.teacher)
            print(f"[info] teacher: {args.teacher} (vocab {len(tea.get_vocab())})", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] teacher tokenizer unavailable ({e}); student-only report", file=sys.stderr)

    audits = [audit_student_vocab(stu)]
    if tea is not None:
        audits.append(audit_teacher_vocab(tea, args.teacher))

    rows, verify = [], None
    if not args.vocab_only:
        caps = collect_captions(args.eval_root, args.babel)
        for k, v in sorted(caps.items()):
            print(f"[info] {k[0]}/{k[1]}: {len(v)} captions", file=sys.stderr)
        if not caps:
            print("[warn] no captions found; pass --eval-root/--babel for part B", file=sys.stderr)
        if args.verify_openclip:
            verify = verify_against_open_clip(stu, caps)
            print(f"[info] open_clip parity: {verify}", file=sys.stderr)
        rows = corpus_stats(caps, stu, tea, ctx=args.ctx)

    payload = {"ctx": args.ctx, "bpe_path": bpe, "teacher": args.teacher,
               "vocab_audit": audits, "corpus_stats": rows, "openclip_parity": verify}
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    md = to_markdown(audits, rows, args.ctx)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)
    print(f"\n[info] wrote {args.out_json} and {args.out_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
