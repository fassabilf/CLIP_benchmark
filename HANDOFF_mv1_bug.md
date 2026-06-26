# HANDOFF — SigLIP2-multilingual student runs collapse

## TL;DR

Tiga run terakhir Habibi yang pakai teacher SigLIP2 dengan data **multilingual**
semua collapse ke skor ≈ random, walau loss training turun normal. Run lama
yang pakai data **EN-heavy** dengan teacher yang sama jalan baik. Bug ada di
suatu tempat di stack multilingual (teacher/tokenizer/data), **bukan**
dataset-nya per se — karena ImageNet EN evaluasi juga collapse, bukan cuma SEA.

## Bukti angka (dari `runs/results/benchmark_aggregate.csv`)

| Run                | Teacher           | Data                          | MEAN ENG | MEAN SEA |
|--------------------|-------------------|-------------------------------|---------:|---------:|
| s3 (e100)          | ViT-B-16-SigLIP2  | v3 (EN-heavy)                 |   0.432  |   0.063  |
| s3_e8              | ViT-B-16-SigLIP2  | v3                            |   0.275  |   0.066  |
| s2 (e32)           | ViT-B-16-SigLIP2  | v2 (EN-heavy)                 |   0.407  |   0.065  |
| **wit_e32**        | ViT-B-16-SigLIP2  | WIT multilingual              |   0.046  |   0.054  |
| **wit_e8**         | ViT-B-16-SigLIP2  | WIT multilingual              |   0.044  |   0.055  |
| **mv1_e32**        | ViT-B-16-SigLIP2  | cultural-ground + WIT + bloom |   0.046  |   0.056  |
| **mv1_e8**         | ViT-B-16-SigLIP2  | cultural-ground + WIT + bloom |   0.045  |   0.053  |
| s1 (e32)           | ViT-B-32 (CLIP)   | v1 (EN)                       |   0.436  |   0.067  |

- Semua run multilingual: ImageNet1k EN top-1 ≈ **0.4-0.5%** (random = 0.1%).
- CVQA stuck di **~0.25** (random untuk 4-way MC) → model tidak belajar
  text-image alignment yang transferable sama sekali.
- Tapi training loss turun (e.g. mv1_e32 step 5439: total 9.62, ckd 1.41,
  contrastive 1.01, ICL 6.23, fd_image 0.97). Jadi optimisasi jalan, hanya
  saja representation tidak transfer.

## Karakter bug

1. **Teacher tetap sama** (`ViT-B-16-SigLIP2`) antara run yang sukses (s2/s3)
   dan run yang collapse (wit/mv1). Jadi bukan teacher checkpoint corrupt.
2. **Student config sama** (`ViT-T-16`, HFTokenizer multilingual). Architecture
   bukan masalah.
3. **EN dan SEA dua-duanya collapse pada wit/mv1.** Kalau bug spesifik bahasa
   SEA, EN seharusnya tetap baik. Jadi bug bersifat **system-wide**, bukan
   per-lokal.
4. **Loss turun, eval nol.** Klasik: representation collapse / shortcut
   learning (semua image → embedding sama, atau semua text → embedding sama).
   Loss intra-batch contrastive bisa kelihatan turun walau embedding degenerate.
5. **Pola identik di wit (1 dataset) dan mv1 (3 dataset)** — jadi bukan bug
   konkatenasi CSV.

## Hipotesis (urutkan dari paling mungkin)

**Konteks penting:** s2/s3 (sukses) dan wit/mv1 (collapse) load via arch
`ViT-T-16` yang sama, dan eval CLIP_benchmark pakai pipeline identik. Eval
pada s2/s3 menghasilkan skor wajar → tokenizer config & eval pipeline bukan
sumber bug. Bug ada di **proses training wit/mv1**.

### H1 — Hyperparam regression di train.sh
Diff `scripts/train.sh` (wit/mv1) vs `scripts/old_train.sh` (v2/v3).
Variabel paling curiga: `ALPHA_FD=2000.0` — kalau di v2/v3 ini lebih kecil,
FD loss bisa mendominasi total loss & menyetir student ke embedding
degenerate. Loss turun (yang kita lihat di wandb), tapi karena yang turun
mainly FD, semantik contrastive ilang.
**Cek:** `diff scripts/train.sh scripts/old_train.sh`.

### H2 — Data quality: caption kosong / image broken
204 baris di `train.csv` punya path `/` (malformed). Mungkin sebagian besar
sample memang loaded tapi caption-nya kosong/nan, atau image gagal di-load
& diganti placeholder. Kalau >X% batch degenerate, contrastive learning
collapse.
**Cek cepat:**
```bash
awk -F',' 'NR>1 && ($2=="" || $1=="" || $1~"^//")' \
  /lustrefs/disk/project/lt200394-thllmV/multilingual-clip-kd/open_clip/train.csv \
  | wc -l
```
Dan sample 20 image path random, cek apakah file-nya ada & valid.

### H3 — Encoding bug saat concat 3 dataset
mv1 train.csv = cultural-ground (58%) + WIT (29%) + bloom (12%). Kalau salah
satu source punya non-UTF8 chars atau quote-handling yang beda, parser CSV
bisa menggeser kolom → caption masuk ke kolom image atau sebaliknya.
**Cek:** sampling 20 baris random, validate img_path exists & caption
non-empty per source.

### H4 — Teacher checkpoint loading inkonsisten
Path `pretrained/siglip2/open_clip_model.safetensors` — apakah file ini
diubah/re-saved antara run v3 dan run wit/mv1? Kalau teacher rusak/ganti,
student "distill" dari noise → collapse.
**Cek:** `ls -la pretrained/siglip2/` dan bandingkan mtime dengan tanggal
mulai run sukses (v3) vs collapse (wit/mv1).

### H5 — train_wit.csv vs train.csv path bug
Ada dua file: `train.csv` (mv1, 174K) dan `train_wit.csv` (50K). train.sh
sekarang reference `train.csv`. Run wit dulu mungkin reference
`train_wit.csv`. Tapi *isi* train_wit.csv mungkin sudah di-overwrite atau
dihapus → run wit_e8/e32 dilatih di data yang berbeda dari yang kita kira.
**Sudah agak tertutup** karena wit & mv1 pola collapse-nya identik (jadi
dataset-nya bukan satu-satunya variabel).

## File yang harus dicek

```
/lustrefs/disk/project/lt200394-thllmV/multilingual-clip-kd/open_clip/
├── scripts/train.sh          # config mv1/wit terbaru
├── scripts/old_train.sh      # konfigurasi v2/v3 yang sukses — diff!
├── experiments/siglip2_kd/
│   ├── clipkd_ViT-T-16_from_ViT-B-16-SigLIP2_v3/                # SUKSES
│   ├── clipkd_ViT-T-16_from_ViT-B-16-SigLIP2_multilingual_v1/    # COLLAPSE
│   │   └── (look for params.txt / config.json / open_clip_pytorch_model.json)
│   └── ...
└── src/open_clip_train/      # kode training (cek di mana tokenizer di-load)
```

## Embedding-collapse smoke check

Untuk konfirmasi degenerate representation (regardless of mana hipotesis
yang benar):

```python
import open_clip, torch
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-T-16",
    pretrained="/project/lt200394-thllmV/multilingual-clip-kd/open_clip/"
               "experiments/siglip2_kd/"
               "clipkd_ViT-T-16_from_ViT-B-16-SigLIP2_multilingual_v1/"
               "checkpoints/epoch_32.pt",
)
model.eval()
tok = open_clip.get_tokenizer("ViT-T-16")

# Encode batch text & image yang berbeda, cek pairwise cosine.
# Kalau collapse, semua cosine ≈ 1.0 (representation degenerate).
texts = tok(["a dog", "kucing oren", "รถยนต์สีแดง", "the eiffel tower",
             "buku terbuka di atas meja"])
with torch.no_grad():
    t = model.encode_text(texts)
    t = t / t.norm(dim=-1, keepdim=True)
print("text cos:", (t @ t.T).numpy().round(3))
```

Kalau hasilnya semua mendekati 1.0 → confirmed embedding collapse.

## Next steps (urutan)

1. **Diff train script:**
   `diff /lustrefs/disk/project/lt200394-thllmV/multilingual-clip-kd/open_clip/scripts/{old_,}train.sh`
   — fokus ke alpha_fd / alpha_ckd / alpha_icl / lr / warmup.
2. **Validate train.csv:** count baris malformed, sample 20 path, cek
   apakah file image ada & caption non-empty.
3. **Run smoke check** di atas — konfirmasi collapse.
4. **Inspect teacher checkpoint mtime** vs tanggal mulai run sukses (v3,
   sebelum Apr 14) vs run collapse (wit dari ~May 18, mv1 dari ~May 24).
