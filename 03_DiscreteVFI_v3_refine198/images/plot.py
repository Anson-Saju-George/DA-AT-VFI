import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# DATA (from your console)
# -------------------------

indices = np.array([
    0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
    10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000,
    18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000,
    26000, 27000, 28000, 29000, 30000, 31000, 32000, 33000,
    34000, 35000, 36000, 37000, 38000, 39000, 40000, 41000,
    42000, 43000, 44000, 45000, 46000, 47000, 48000, 49000,
    50000, 51000, 52000
])

psnr = np.array([
    40.89, 32.10, 35.72, 34.60, 35.24, 29.48, 36.42, 35.11,
    39.84, 37.66, 35.44, 32.50, 33.62, 42.65, 44.44, 35.82,
    37.15, 43.83, 42.31, 29.26, 36.63, 30.36, 34.96, 20.73,
    41.27, 32.25, 36.67, 29.58, 35.49, 34.00, 37.21, 39.00,
    33.14, 35.96, 30.09, 15.29, 28.85, 31.87, 27.69, 31.64,
    36.71, 37.22, 58.42, 34.08, 43.47, 45.77, 50.53, 34.10,
    33.41, 31.07, 8.77, 13.54, 12.18
])

ssim = np.array([
    0.9791, 0.9404, 0.9719, 0.9637, 0.9647, 0.9573, 0.9597,
    0.9370, 0.9746, 0.9578, 0.9695, 0.9503, 0.9306, 0.9815,
    0.9889, 0.9370, 0.9436, 0.9818, 0.9737, 0.8955, 0.9645,
    0.8686, 0.9220, 0.6837, 0.9770, 0.9351, 0.9503, 0.9063,
    0.9479, 0.9531, 0.9473, 0.9671, 0.9192, 0.9488, 0.9077,
    0.3897, 0.8701, 0.9488, 0.8561, 0.9204, 0.9638, 0.9831,
    0.9987, 0.9420, 0.9911, 0.9890, 0.9978, 0.9524, 0.8744,
    0.8584, 0.0675, 0.1062, 0.1420
])

# -------------------------
# STYLE CONFIG
# -------------------------
DPI = 300
FIGSIZE = (7, 4)
WINDOW = 5
FAIL_THRESHOLD = 20

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": DPI
})

def save_fig(name):
    plt.tight_layout(pad=1.2)
    plt.savefig(name, dpi=DPI, bbox_inches="tight")
    plt.close()

mean_psnr = psnr.mean()
mean_ssim = ssim.mean()

# -------------------------
# 1. PSNR with Mean Line
# -------------------------
plt.figure(figsize=FIGSIZE)
plt.plot(indices, psnr, marker="o", linewidth=1, label="PSNR")
plt.axhline(mean_psnr, linestyle="--", linewidth=2, label=f"Mean = {mean_psnr:.2f} dB")
plt.title("PSNR vs Sample Index (with Mean)")
plt.xlabel("Sample Index")
plt.ylabel("PSNR (dB)")
plt.legend()
plt.grid(True, alpha=0.3)
save_fig("fig_psnr_with_mean.png")

# -------------------------
# 2. SSIM with Mean Line
# -------------------------
plt.figure(figsize=FIGSIZE)
plt.plot(indices, ssim, marker="o", linewidth=1, label="SSIM")
plt.axhline(mean_ssim, linestyle="--", linewidth=2, label=f"Mean = {mean_ssim:.3f}")
plt.title("SSIM vs Sample Index (with Mean)")
plt.xlabel("Sample Index")
plt.ylabel("SSIM")
plt.ylim(0, 1.0)
plt.legend()
plt.grid(True, alpha=0.3)
save_fig("fig_ssim_with_mean.png")

# -------------------------
# 3. Rolling PSNR
# -------------------------
rolling = np.convolve(psnr, np.ones(WINDOW) / WINDOW, mode="valid")

plt.figure(figsize=FIGSIZE)
plt.plot(indices[WINDOW - 1:], rolling, linewidth=2)
plt.title(f"Rolling PSNR Trend (Window = {WINDOW})")
plt.xlabel("Sample Index")
plt.ylabel("PSNR (dB)")
plt.grid(True, alpha=0.3)
save_fig("fig_psnr_rolling.png")

# -------------------------
# 4. Mean Metrics Bar Chart
# -------------------------
plt.figure(figsize=(5, 4))
plt.bar(["Mean PSNR (dB)", "Mean SSIM"], [mean_psnr, mean_ssim])
plt.title("Mean Evaluation Metrics (WMS)")
plt.ylabel("Value")
plt.grid(True, axis="y", alpha=0.3)
save_fig("fig_mean_metrics.png")

# -------------------------
# 5. Failure Rate Curve
# -------------------------
fails = (psnr < FAIL_THRESHOLD).astype(np.float32)
failure_rate = np.cumsum(fails) / (np.arange(len(psnr)) + 1)

plt.figure(figsize=FIGSIZE)
plt.plot(indices, failure_rate, linewidth=2)
plt.title(f"Cumulative Failure Rate (PSNR < {FAIL_THRESHOLD} dB)")
plt.xlabel("Sample Index")
plt.ylabel("Failure Rate")
plt.ylim(0, 1.0)
plt.grid(True, alpha=0.3)
save_fig("fig_failure_rate.png")

# -------------------------
# SUMMARY TEXT
# -------------------------
fail_pct = failure_rate[-1] * 100

with open("eval_summary.txt", "w") as f:
    f.write("Evaluation Summary — DiscreteVFI_v3_refine198 (WMS)\n")
    f.write("===============================================\n")
    f.write(f"Mean PSNR: {mean_psnr:.3f} dB\n")
    f.write(f"Mean SSIM: {mean_ssim:.4f}\n")
    f.write(f"Failure Threshold: {FAIL_THRESHOLD} dB\n")
    f.write(f"Final Failure Rate: {fail_pct:.2f}%\n")

print("Saved figures in current directory:")
print(" - fig_psnr_with_mean.png")
print(" - fig_ssim_with_mean.png")
print(" - fig_psnr_rolling.png")
print(" - fig_mean_metrics.png")
print(" - fig_failure_rate.png")
print(" - eval_summary.txt")
