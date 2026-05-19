"""
Digital Watermarking + JPEG Compression — From Scratch
=======================================================
Tugas:
  - Tambahkan watermark (binary atau acak) pada foto wajah
  - Kompres dengan JPEG (implementasi sendiri: DCT + kuantisasi)
  - Evaluasi NC (Normalized Correlation) pada berbagai Quality Factor
  - Temukan QF di mana watermark tidak bisa diekstrak

Dependensi: numpy, matplotlib, Pillow (hanya untuk baca/tulis file gambar)
            Kompresi JPEG sepenuhnya dari scratch.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

try:
    from PIL import Image
    _PIL = True
except ImportError:
    _PIL = False


# ============================================================
# TABEL KUANTISASI JPEG (standar ISO/IEC 10918-1)
# ============================================================

_LUMA_Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99],
], dtype=np.float64)

_CHROMA_Q50 = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float64)


def get_quant_table(qf: int, channel: str = 'luma') -> np.ndarray:
    """
    Skala tabel kuantisasi berdasarkan Quality Factor (1–100).

    Rumus skala (dari spesifikasi JPEG):
      if qf < 50 : scale = 5000 / qf
      else       : scale = 200 - 2*qf
      table = floor((base * scale + 50) / 100), clamp ke [1, 255]
    """
    qf = int(np.clip(qf, 1, 100))
    scale = 5000.0 / qf if qf < 50 else 200.0 - 2.0 * qf
    base  = _LUMA_Q50 if channel == 'luma' else _CHROMA_Q50
    table = np.floor((base * scale + 50.0) / 100.0)
    return np.clip(table, 1.0, 255.0)


# ============================================================
# DCT-II 2D — FROM SCRATCH
# ============================================================

def _build_dct_matrix(N: int = 8) -> np.ndarray:
    """
    Bangun matriks DCT-II ortogonal N×N.

    Definisi:
      D[k,n] = sqrt(1/N)          , k = 0
             = sqrt(2/N)*cos(pi*k*(2n+1)/(2N)), k > 0

    Sifat: D adalah ortogonal, sehingga D^{-1} = D^T.
    """
    D = np.zeros((N, N), dtype=np.float64)
    for k in range(N):
        for n in range(N):
            if k == 0:
                D[k, n] = np.sqrt(1.0 / N)
            else:
                D[k, n] = np.sqrt(2.0 / N) * np.cos(np.pi * k * (2*n + 1) / (2.0 * N))
    return D


# Precompute 8×8 DCT matrix sekali saja
_D  = _build_dct_matrix(8)
_DT = _D.T   # inverse = transpose (karena ortogonal)


def dct2(block: np.ndarray) -> np.ndarray:
    """2D DCT-II blok 8×8:  F = D · f · D^T"""
    return _D @ block @ _DT


def idct2(F: np.ndarray) -> np.ndarray:
    """2D IDCT-II blok 8×8:  f = D^T · F · D"""
    return _DT @ F @ _D


# ============================================================
# KONVERSI RUANG WARNA
# ============================================================

def rgb_to_ycbcr(img: np.ndarray) -> np.ndarray:
    """
    RGB uint8 [H,W,3] → YCbCr float64 [H,W,3]

    Rumus ITU-R BT.601:
      Y  =  0.299 R + 0.587 G + 0.114 B
      Cb = -0.169 R - 0.331 G + 0.500 B + 128
      Cr =  0.500 R - 0.419 G - 0.081 B + 128
    """
    R = img[:, :, 0].astype(np.float64)
    G = img[:, :, 1].astype(np.float64)
    B = img[:, :, 2].astype(np.float64)
    Y  =  0.29900*R + 0.58700*G + 0.11400*B
    Cb = -0.16874*R - 0.33126*G + 0.50000*B + 128.0
    Cr =  0.50000*R - 0.41869*G - 0.08131*B + 128.0
    return np.stack([Y, Cb, Cr], axis=2)


def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    """YCbCr float64 [H,W,3] → RGB uint8 [H,W,3]"""
    Y  = ycbcr[:, :, 0]
    Cb = ycbcr[:, :, 1] - 128.0
    Cr = ycbcr[:, :, 2] - 128.0
    R  = Y + 1.40200 * Cr
    G  = Y - 0.34414 * Cb - 0.71414 * Cr
    B  = Y + 1.77200 * Cb
    return np.clip(np.stack([R, G, B], axis=2), 0, 255).astype(np.uint8)


# ============================================================
# JPEG SIMULATE — FROM SCRATCH
# ============================================================

def jpeg_simulate(img_rgb: np.ndarray, qf: int) -> np.ndarray:
    """
    Simulasi kompresi-dekompresi JPEG dari scratch.

    Pipeline:
      1. RGB → YCbCr
      2. Pad gambar ke kelipatan 8
      3. Reshape seluruh channel menjadi array blok (nh, nw, 8, 8)
      4. DCT-II 2D batch: F = D · blocks · D^T  (numpy broadcasting)
      5. Kuantisasi batch: Fq = round(F / Q) * Q  ← lossy step
      6. IDCT-II 2D batch: recon = D^T · Fq · D
      7. Reshape balik, crop padding, clip ke [0,255]
      8. YCbCr → RGB
    """
    ycbcr = rgb_to_ycbcr(img_rgb)
    H, W  = ycbcr.shape[:2]
    pH    = H + (-H % 8)
    pW    = W + (-W % 8)
    nh, nw = pH // 8, pW // 8
    out   = np.zeros((H, W, 3), dtype=np.float64)

    for c in range(3):
        qtable = get_quant_table(qf, 'luma' if c == 0 else 'chroma')
        ch     = ycbcr[:, :, c]
        padded = np.pad(ch, ((0, pH - H), (0, pW - W)), mode='edge') - 128.0

        # Reshape ke (nh, nw, 8, 8): semua blok sekaligus
        blocks = padded.reshape(nh, 8, nw, 8).transpose(0, 2, 1, 3)

        # Batch DCT: _D @ blocks @ _DT  — numpy broadcast ke (nh, nw, 8, 8)
        F  = _D @ blocks @ _DT

        # Kuantisasi + dekuantisasi (satu operasi, semua blok)
        Fq = np.round(F / qtable) * qtable

        # Batch IDCT
        recon_blocks = _DT @ Fq @ _D

        # Reshape balik ke (pH, pW)
        recon = recon_blocks.transpose(0, 2, 1, 3).reshape(pH, pW)
        out[:, :, c] = np.clip(recon[:H, :W] + 128.0, 0.0, 255.0)

    return ycbcr_to_rgb(out)


# ============================================================
# WATERMARK
# ============================================================

def make_watermark(H: int, W: int, kind: str = 'binary', seed: int = 42) -> np.ndarray:
    """
    Buat watermark berukuran H×W.

    kind='binary' : citra biner {0,1} lalu dipetakan ke {-1,+1} (BPSK)
    kind='random' : derau Gaussian (µ=0, σ=1)
    """
    rng = np.random.default_rng(seed)
    if kind == 'binary':
        bits = rng.integers(0, 2, size=(H, W)).astype(np.float64)
        return bits * 2.0 - 1.0          # {0,1} → {-1,+1}
    else:
        noise = rng.standard_normal((H, W))
        return noise / (noise.std() + 1e-10)   # normalisasi σ=1


def embed(img: np.ndarray, wm: np.ndarray, alpha: float) -> np.ndarray:
    """
    Penyisipan watermark aditif:
      I_w[x,y] = clip(I[x,y] + alpha * w[x,y], 0, 255)
    Watermark disisipkan pada ketiga kanal warna.
    """
    wm3 = np.stack([wm, wm, wm], axis=2)
    out = img.astype(np.float64) + alpha * wm3
    return np.clip(out, 0, 255).astype(np.uint8)


def extract(original: np.ndarray, test: np.ndarray, alpha: float) -> np.ndarray:
    """
    Ekstraksi watermark non-blind (membutuhkan gambar asli):
      w_est = (I_compressed - I_original) / alpha

    Rumus ini berlaku karena:
      I_compressed ≈ I_original + alpha*w + noise_JPEG
      → w_est = w + noise_JPEG/alpha

    Saat noise_JPEG besar (QF rendah), NC turun dan watermark hilang.
    """
    diff = test.astype(np.float64) - original.astype(np.float64)
    # Rata-rata ketiga kanal untuk estimasi lebih baik
    return diff.mean(axis=2) / alpha


def normalized_correlation(w1: np.ndarray, w2: np.ndarray) -> float:
    """
    Normalized Correlation (NC) ∈ [-1, 1].

    NC = (w1 - µ1)·(w2 - µ2) / (||w1 - µ1|| · ||w2 - µ2||)

    NC ≈ 1  → watermark terekstrak dengan baik
    NC ≈ 0  → watermark hilang / tidak berkorelasi
    """
    a = w1.ravel() - w1.mean()
    b = w2.ravel() - w2.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    """PSNR (Peak Signal-to-Noise Ratio) dalam dB."""
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else 10.0 * np.log10(255.0**2 / mse)


# ============================================================
# LOAD GAMBAR
# ============================================================

def load_face(path: str = 'face.jpg') -> np.ndarray:
    """
    Load foto wajah dari file. Jika tidak ada, buat gambar sintetis.
    Resize ke maks 512px agar cepat.
    """
    if _PIL and os.path.exists(path):
        print(f"[+] Loaded '{path}'")
        img = np.array(Image.open(path).convert('RGB'))
        h, w = img.shape[:2]
        if max(h, w) > 512:
            scale = 512 / max(h, w)
            nh, nw = int(h * scale), int(w * scale)
            # Paksa ke kelipatan 8 supaya tidak ada sisa padding aneh
            nh = (nh // 8) * 8
            nw = (nw // 8) * 8
            img = np.array(Image.fromarray(img).resize((nw, nh), Image.LANCZOS))
            print(f"    Resized to {nw}×{nh}")
        return img

    print(f"[!] '{path}' tidak ditemukan — menggunakan gambar sintetis.")
    img = np.full((256, 256, 3), [210, 175, 145], dtype=np.uint8)
    img[70:200, 70:200]    = [225, 185, 155]   # wajah
    img[88:112, 83:108]    = [40, 35, 30]       # mata kiri
    img[88:112, 148:173]   = [40, 35, 30]       # mata kanan
    img[153:170, 98:160]   = [180, 80, 80]      # mulut
    img[60:75,  80:180]    = [180, 140, 100]    # dahi
    return img


# ============================================================
# MAIN — EVALUASI
# ============================================================

if __name__ == '__main__':
    # ---- Parameter ----
    ALPHA     = 20.0        # kekuatan penyisipan
    WM_KIND   = 'binary'    # 'binary' atau 'random'
    SEED      = 42
    NC_THRESH = 0.5         # batas NC untuk "masih bisa diekstrak"
    QFS       = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

    print("=" * 60)
    print("  Watermarking + JPEG Compression — From Scratch")
    print("=" * 60)

    # Load & watermark
    original = load_face('face.jpg')
    H, W     = original.shape[:2]
    print(f"Ukuran gambar : {W}×{H} px")

    wm     = make_watermark(H, W, kind=WM_KIND, seed=SEED)
    marked = embed(original, wm, alpha=ALPHA)

    wm_psnr = psnr(original, marked)
    print(f"Jenis watermark    : {WM_KIND}")
    print(f"Alpha (kekuatan)   : {ALPHA}")
    print(f"PSNR watermarked   : {wm_psnr:.2f} dB  (makin tinggi = makin tak kasat mata)")

    # ---- Loop evaluasi QF ----
    print("\n{:>5}  {:>8}  {:>14}  {}".format("QF", "NC", "PSNR_img (dB)", "Status"))
    print("-" * 52)

    nc_vals        = []
    psnr_img_vals  = []
    compressed_at  = {}

    for qf in QFS:
        comp   = jpeg_simulate(marked, qf)
        wm_est = extract(original, comp, alpha=ALPHA)
        nc_val = normalized_correlation(wm, wm_est)
        p      = psnr(original, comp)

        nc_vals.append(nc_val)
        psnr_img_vals.append(p)
        compressed_at[qf] = comp

        status = "OK  (terekstrak)" if nc_val > NC_THRESH else "HILANG"
        print(f"{qf:>5}  {nc_val:>8.4f}  {p:>14.2f}  {status}")

    # Temukan QF kritis
    first_lost = next((qf for qf, nc_v in zip(QFS, nc_vals) if nc_v <= NC_THRESH), None)
    print()
    if first_lost:
        print(f"=> Watermark tidak dapat diekstrak pada QF <= {first_lost}  (NC <= {NC_THRESH})")
    else:
        print("=> Watermark tetap terekstrak pada semua QF yang diuji.")

    # ============================================================
    # VISUALISASI
    # ============================================================
    SHOWCASE = [100, 70, 40, 10]   # QF yang ditampilkan gambar & ekstraksi

    fig = plt.figure(figsize=(20, 13))
    fig.suptitle(
        f'Evaluasi Watermarking — JPEG From Scratch\n'
        f'(watermark={WM_KIND}, α={ALPHA}, NC-threshold={NC_THRESH})',
        fontsize=13, fontweight='bold'
    )

    # ---- Baris 1: citra asli, watermark, watermarked, selisih ----
    ax = fig.add_subplot(3, 4, 1)
    ax.imshow(original); ax.set_title('Citra Asli'); ax.axis('off')

    wm_display = (wm + 1) / 2   # petakan {-1,+1} ke [0,1] untuk tampilan
    ax = fig.add_subplot(3, 4, 2)
    ax.imshow(wm_display, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Watermark ({WM_KIND})'); ax.axis('off')

    ax = fig.add_subplot(3, 4, 3)
    ax.imshow(marked); ax.set_title(f'Watermarked (α={ALPHA})'); ax.axis('off')

    diff_vis = np.clip(
        (marked.astype(np.int32) - original.astype(np.int32)) * 8 + 128, 0, 255
    ).astype(np.uint8)
    ax = fig.add_subplot(3, 4, 4)
    ax.imshow(diff_vis); ax.set_title('Selisih ×8 (visibilitas)'); ax.axis('off')

    # ---- Baris 2: citra setelah kompresi JPEG ----
    for k, qf in enumerate(SHOWCASE):
        nc_v = nc_vals[QFS.index(qf)]
        ax   = fig.add_subplot(3, 4, 5 + k)
        ax.imshow(compressed_at[qf])
        status_str = f'NC={nc_v:.3f} ✓' if nc_v > NC_THRESH else f'NC={nc_v:.3f} ✗'
        ax.set_title(f'JPEG QF={qf}  {status_str}'); ax.axis('off')

    # ---- Baris 3: watermark terekstrak + plot NC ----
    for k, qf in enumerate(SHOWCASE[:3]):
        wm_est = extract(original, compressed_at[qf], alpha=ALPHA)
        ax     = fig.add_subplot(3, 4, 9 + k)
        ax.imshow(wm_est, cmap='gray')
        ax.set_title(f'Ekstrak WM (QF={qf})'); ax.axis('off')

    # Plot NC vs QF
    ax = fig.add_subplot(3, 4, 12)
    ax.plot(QFS, nc_vals, 'b-o', linewidth=2.5, markersize=8, label='NC')
    ax.axhline(NC_THRESH, color='red', linestyle='--', linewidth=1.8,
               label=f'Threshold={NC_THRESH}')
    if first_lost:
        ax.axvline(first_lost, color='orange', linestyle=':', linewidth=2,
                   label=f'Hilang QF={first_lost}')
    ax.set_xlabel('Quality Factor (QF)')
    ax.set_ylabel('Normalized Correlation (NC)')
    ax.set_title('Ketahanan Watermark vs QF')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.35)
    ax.invert_xaxis()
    ax.set_ylim(-0.15, 1.15)
    ax.set_xlim(max(QFS) + 5, min(QFS) - 5)

    plt.tight_layout()
    out = 'watermark_evaluation.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\nHasil disimpan ke '{out}'")
    plt.show()
