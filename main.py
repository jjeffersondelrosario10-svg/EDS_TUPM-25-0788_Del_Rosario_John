"""
============================================================
 Engineering Data Systems Pipeline
 Topic  : RMS-01 — Wind Turbine Gearbox Vibration Analytics
 Dataset: Wind Turbine SCADA Dataset (Yalova, Turkey, 2018)
          Kaggle: berkerisen/wind-turbine-scada-dataset
          File  : T1.csv

 Author    : John Jefferson DA. Del Rosario
 Student ID: TUPM-25-0788
 Program   : BS Mechanical Engineering
 Course    : Computer Programming
 School    : Technological University of the Philippines, Manila
 A.Y.      : 2026

 Unique Filter Logic (Birthday: January 10, 2007):
   Month  = January (birth month)
   Wind Direction sector = 0 to 36 degrees
     (birth day 10 x 3.6 degrees per day = sector 0-36)
   This combination produces a unique data slice.
============================================================
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

# Configuration
DATA_PATH    = "data/T1.csv"
CLEANED_PATH = "data/dataset_cleaned.csv"
OUTPUT_DIR   = "outputs"

FILTER_MONTH  = 1
FILTER_DIR_LO = 0.0
FILTER_DIR_HI = 36.0
RANDOM_SEED   = 2007

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)

COL_TIME  = "Date/Time"
COL_POWER = "LV ActivePower (kW)"
COL_WIND  = "Wind Speed (m/s)"
COL_THEOR = "Theoretical_Power_Curve (KWh)"
COL_DIR   = "Wind Direction (deg)"
NUM_COLS  = [COL_POWER, COL_WIND, COL_THEOR, COL_DIR]


class WindTurbinePipeline:
    """
    OOP pipeline for Wind Turbine SCADA Analytics (RMS-01).

    Modules:
      1. ingest()    - load CSV + unique birthday filter
      2. clean()     - remove dupes, nulls, corrupted values
      3. analyze()   - NumPy statistical analytics engine
      4. visualize() - 5 static plots + 2 animated GIFs
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_df    = None
        self.clean_df  = None
        self.stats     = {}
        print("=" * 62)
        print("  RMS-01 | Wind Turbine SCADA Analytics Pipeline")
        print("  Del Rosario, J.J.D.A. | TUPM-25-0788 | ME")
        print("=" * 62)

    # MODULE 1: DATA INGESTION
    def ingest(self):
        print("\n[MODULE 1] DATA INGESTION")
        print("-" * 40)
        try:
            df = pd.read_csv(self.data_path)
            # Rename Wind Direction column to avoid special chars
            df.columns = [c.replace("(°)", "(deg)") for c in df.columns]
            df[COL_TIME] = pd.to_datetime(df[COL_TIME])
            total = len(df)
            print(f"  Records loaded    : {total:,}")
            print(f"  Columns           : {list(df.columns)}")

            # Unique filter
            dir_num    = pd.to_numeric(df[COL_DIR], errors="coerce")
            month_mask = df[COL_TIME].dt.month == FILTER_MONTH
            dir_mask   = (dir_num >= FILTER_DIR_LO) & (dir_num < FILTER_DIR_HI)
            df = df[month_mask & dir_mask].copy().reset_index(drop=True)

            print(f"\n  Unique Filter Applied:")
            print(f"    Month = January | Direction = {FILTER_DIR_LO}deg to {FILTER_DIR_HI}deg")
            print(f"  Records after filter : {len(df):,} ({100*len(df)/total:.2f}%)")

            if len(df) < 10:
                raise ValueError("Filter returned too few rows.")

            self.raw_df = df
            return self.raw_df

        except FileNotFoundError:
            print(f"  [ERROR] {self.data_path} not found.")
            print("  Download T1.csv from Kaggle:")
            print("  kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset")
            sys.exit(1)
        except Exception as e:
            print(f"  [ERROR] Ingestion failed: {e}")
            sys.exit(1)

    # MODULE 2: DATA CLEANING
    def clean(self):
        print("\n[MODULE 2] DATA CLEANING")
        print("-" * 40)
        try:
            df = self.raw_df.copy()
            n0 = len(df)

            # Step 1: Remove duplicates
            dupes = df.duplicated().sum()
            df = df.drop_duplicates().reset_index(drop=True)
            print(f"  Duplicates removed           : {dupes}")

            # Step 2: Coerce numeric columns; invalid -> NaN
            corrupt = 0
            for col in NUM_COLS:
                before = df[col].isnull().sum()
                df[col] = pd.to_numeric(df[col], errors="coerce")
                corrupt += max(0, df[col].isnull().sum() - before)
            print(f"  Corrupted values -> NaN      : {corrupt}")

            # Step 3: Drop rows with NaN in key columns
            nulls = df[NUM_COLS].isnull().sum().sum()
            df = df.dropna(subset=NUM_COLS).reset_index(drop=True)
            print(f"  Null rows removed            : {nulls}")

            # Step 4: Physical range validation
            rules = {
                COL_WIND  : (0,   25),
                COL_POWER : (-50, 2100),
                COL_THEOR : (0,   2100),
                COL_DIR   : (0,   360),
            }
            oor = 0
            for col, (lo, hi) in rules.items():
                mask = (df[col] < lo) | (df[col] > hi)
                oor += mask.sum()
                df = df[~mask]
            df = df.reset_index(drop=True)
            print(f"  Out-of-range removed         : {oor}")

            # Step 5: Derived features
            df["Power_Deviation_kW"] = (df[COL_POWER] - df[COL_THEOR]).round(4)
            df["Capacity_Factor"]    = (df[COL_POWER] / 2000.0).round(6)

            print(f"  Clean records retained       : {len(df):,} (from {n0:,})")

            df.to_csv(CLEANED_PATH, index=False)
            print(f"  Saved -> {CLEANED_PATH}")

            self.clean_df = df
            return self.clean_df

        except Exception as e:
            print(f"  [ERROR] Cleaning failed: {e}")
            sys.exit(1)

    # MODULE 3: STATISTICAL ANALYSIS
    def analyze(self):
        print("\n[MODULE 3] STATISTICAL ANALYSIS")
        print("-" * 40)
        try:
            df  = self.clean_df
            out = {}

            def _stats(arr, label):
                arr = arr[~np.isnan(arr)]
                q1, q3 = np.percentile(arr, [25, 75])
                iqr = q3 - q1
                return {
                    "label"   : label,
                    "n"       : len(arr),
                    "mean"    : float(np.mean(arr)),
                    "median"  : float(np.median(arr)),
                    "std"     : float(np.std(arr, ddof=1)),
                    "var"     : float(np.var(arr, ddof=1)),
                    "min"     : float(np.min(arr)),
                    "max"     : float(np.max(arr)),
                    "range"   : float(np.ptp(arr)),
                    "q1"      : float(q1),
                    "q3"      : float(q3),
                    "iqr"     : float(iqr),
                    "skew"    : float(stats.skew(arr)),
                    "kurt"    : float(stats.kurtosis(arr)),
                    "lo_fence": float(q1 - 1.5 * iqr),
                    "hi_fence": float(q3 + 1.5 * iqr),
                }

            # 3A: Descriptive stats
            for col in [COL_WIND, COL_POWER, COL_THEOR, "Power_Deviation_kW"]:
                arr_np = np.array(df[col], dtype=np.float64)
                s = _stats(arr_np, col)
                out[col] = s
                print(f"\n  [{col}]")
                for k in ["mean", "median", "std", "var", "skew", "kurt"]:
                    print(f"    {k:<8} = {s[k]:>12.4f}")

            # 3B: Outlier detection
            pwr_arr = np.array(df[COL_POWER], dtype=np.float64)
            sp = out[COL_POWER]
            omask = (pwr_arr < sp["lo_fence"]) | (pwr_arr > sp["hi_fence"])
            out["outliers"] = {
                "count": int(omask.sum()),
                "pct"  : float(100 * omask.sum() / len(pwr_arr)),
                "mask" : omask,
            }
            print(f"\n  [Outliers - {COL_POWER} IQR]")
            print(f"    Count = {out['outliers']['count']}  ({out['outliers']['pct']:.2f}%)")
            print(f"    Fences = [{sp['lo_fence']:.2f}, {sp['hi_fence']:.2f}]")

            # 3C: Pearson correlation
            corr_cols = [COL_WIND, COL_POWER, COL_THEOR,
                         "Power_Deviation_kW", "Capacity_Factor"]
            mat = np.corrcoef(df[corr_cols].values.T)
            out["corr"] = {"matrix": mat, "cols": corr_cols}
            wi = corr_cols.index(COL_WIND)
            pi = corr_cols.index(COL_POWER)
            ti = corr_cols.index(COL_THEOR)
            out["r_wind_power"]  = float(mat[wi, pi])
            out["r_theor_power"] = float(mat[ti, pi])
            print(f"\n  [Pearson Correlations]")
            print(f"    r(Wind Speed, LV Power)  = {out['r_wind_power']:.6f}")
            print(f"    r(Theoretical, LV Power) = {out['r_theor_power']:.6f}")

            # 3D: Comparative analysis
            ws_med  = float(np.median(df[COL_WIND]))
            lo_mask = df[COL_WIND] <= ws_med
            lo_pwr  = np.array(df.loc[lo_mask,  COL_POWER], dtype=np.float64)
            hi_pwr  = np.array(df.loc[~lo_mask, COL_POWER], dtype=np.float64)
            out["comparative"] = {
                "ws_median": ws_med,
                "low_wind" : _stats(lo_pwr, "Low Wind"),
                "high_wind": _stats(hi_pwr, "High Wind"),
            }
            print(f"\n  [Comparative - Low vs High Wind]")
            print(f"    Median cutoff = {ws_med:.4f} m/s")
            for g in ["low_wind", "high_wind"]:
                v = out["comparative"][g]
                print(f"    {g:<10}: n={v['n']:>4}, mean={v['mean']:>8.2f} kW, std={v['std']:>7.2f}")

            # 3E: Capacity factor
            cf = np.array(df["Capacity_Factor"], dtype=np.float64)
            out["cf_mean"] = float(np.mean(cf))
            print(f"\n  Mean Capacity Factor = {out['cf_mean']:.4f} ({out['cf_mean']*100:.2f}%)")

            self.stats = out
            return self.stats

        except Exception as e:
            print(f"  [ERROR] Analysis failed: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)

    # MODULE 4: VISUALIZATION
    def visualize(self):
        print("\n[MODULE 4] VISUALIZATION")
        print("-" * 40)
        df    = self.clean_df
        s     = self.stats
        saved = []

        DARK   = "#0d1117"
        PANEL  = "#161b22"
        BLUE   = "#58a6ff"
        RED    = "#f78166"
        GREEN  = "#3fb950"
        ORANGE = "#ffa657"
        GRAY   = "#8b949e"
        BORDER = "#30363d"
        LGBG   = "#21262d"

        def _style(fig, axs):
            fig.patch.set_facecolor(DARK)
            for ax in (axs if hasattr(axs, "__iter__") else [axs]):
                ax.set_facecolor(PANEL)
                ax.tick_params(colors=GRAY, labelsize=9)
                for sp in ax.spines.values():
                    sp.set_edgecolor(BORDER)

        # PLOT 1: Histogram + KDE
        arr_pwr = np.array(df[COL_POWER], dtype=np.float64)
        fig, ax = plt.subplots(figsize=(9, 5))
        _style(fig, ax)
        ax.hist(arr_pwr, bins=60, color=BLUE, edgecolor=DARK,
                alpha=0.80, label="LV ActivePower")
        ax2 = ax.twinx()
        ax2.set_facecolor(PANEL)
        kde_x = np.linspace(arr_pwr.min(), arr_pwr.max(), 500)
        kde   = stats.gaussian_kde(arr_pwr)
        ax2.plot(kde_x, kde(kde_x), color=RED, lw=2.5, label="KDE")
        ax2.set_ylabel("Density", color=RED, fontsize=10)
        ax2.tick_params(colors=RED)
        ax.axvline(s[COL_POWER]["mean"],   color=ORANGE, ls="--", lw=1.8,
                   label=f"Mean = {s[COL_POWER]['mean']:.1f} kW")
        ax.axvline(s[COL_POWER]["median"], color=GREEN,  ls="--", lw=1.8,
                   label=f"Median = {s[COL_POWER]['median']:.1f} kW")
        ax.set_title("LV Active Power Distribution - January, Sector 0-36 deg\nYalova Wind Turbine, Turkey (2018) | T1.csv",
                     color="white", fontsize=11, pad=10)
        ax.set_xlabel("LV ActivePower (kW)", color=GRAY, fontsize=10)
        ax.set_ylabel("Frequency",           color=GRAY, fontsize=10)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2, facecolor=LGBG, labelcolor="white",
                  fontsize=8, loc="upper right")
        p = f"{OUTPUT_DIR}/plot1_power_histogram_kde.png"
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=DARK)
        plt.close(); saved.append(p); print(f"  Saved -> {p}")

        # PLOT 2: Boxplot Low vs High Wind
        ws_med = s["comparative"]["ws_median"]
        lo_pwr = df.loc[df[COL_WIND] <= ws_med, COL_POWER].values
        hi_pwr = df.loc[df[COL_WIND] >  ws_med, COL_POWER].values
        fig, ax = plt.subplots(figsize=(8, 5))
        _style(fig, ax)
        bp = ax.boxplot(
            [lo_pwr, hi_pwr], patch_artist=True,
            labels=[f"Low Wind\n(<= {ws_med:.2f} m/s)",
                    f"High Wind\n(> {ws_med:.2f} m/s)"],
            medianprops=dict(color=RED, lw=2.2),
            whiskerprops=dict(color=GRAY), capprops=dict(color=GRAY),
            flierprops=dict(marker="o", markerfacecolor=ORANGE,
                            markersize=3, alpha=0.4),
        )
        for patch, color in zip(bp["boxes"], ["#1f6feb", GREEN]):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        lv = s["comparative"]["low_wind"]
        hv = s["comparative"]["high_wind"]
        ann = (f"Low  mean={lv['mean']:.1f} kW  std={lv['std']:.1f}\n"
               f"High mean={hv['mean']:.1f} kW  std={hv['std']:.1f}")
        ax.text(0.98, 0.97, ann, transform=ax.transAxes,
                va="top", ha="right", fontsize=8, color=GRAY,
                bbox=dict(facecolor=LGBG, edgecolor=BORDER,
                          boxstyle="round", alpha=0.8))
        ax.set_title("LV Active Power: Low vs High Wind Speed Groups",
                     color="white", fontsize=11, pad=10)
        ax.set_ylabel("LV ActivePower (kW)", color=GRAY, fontsize=10)
        p = f"{OUTPUT_DIR}/plot2_boxplot_wind_groups.png"
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=DARK)
        plt.close(); saved.append(p); print(f"  Saved -> {p}")

        # PLOT 3: Correlation Heatmap
        cmat = s["corr"]["matrix"]
        clabels = ["Wind Speed", "LV Power", "Theor. Curve",
                   "Power Dev.", "Capacity\nFactor"]
        n = len(clabels)
        fig, ax = plt.subplots(figsize=(8, 6))
        _style(fig, ax)
        cmap = LinearSegmentedColormap.from_list("div", [RED, PANEL, BLUE])
        im   = ax.imshow(cmat, cmap=cmap, vmin=-1, vmax=1)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.outline.set_edgecolor(BORDER)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="white")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(clabels, color=GRAY, fontsize=8,
                           rotation=30, ha="right")
        ax.set_yticklabels(clabels, color=GRAY, fontsize=8)
        for i in range(n):
            for j in range(n):
                v = cmat[i, j]
                c = "white" if abs(v) < 0.55 else DARK
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8.5, color=c, fontweight="bold")
        ax.set_title("Pearson Correlation Matrix - January, 0-36 deg Sector",
                     color="white", fontsize=11, pad=12)
        p = f"{OUTPUT_DIR}/plot3_correlation_heatmap.png"
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=DARK)
        plt.close(); saved.append(p); print(f"  Saved -> {p}")

        # PLOT 4: Scatter Wind Speed vs Power + Theoretical Curve
        fig, ax = plt.subplots(figsize=(9, 5))
        _style(fig, ax)
        ax.scatter(df[COL_WIND], df[COL_POWER],
                   s=7, alpha=0.45, color=BLUE, label="Actual Power")
        df_s = df.sort_values(COL_WIND)
        ax.plot(df_s[COL_WIND], df_s[COL_THEOR],
                color=RED, lw=1.8, alpha=0.7, label="Theoretical Curve")
        ax.text(0.03, 0.96,
                f"r(Wind, Power) = {s['r_wind_power']:.4f}",
                transform=ax.transAxes, color=ORANGE, fontsize=9,
                va="top", bbox=dict(facecolor=LGBG, edgecolor=BORDER,
                                    boxstyle="round", alpha=0.85))
        ax.set_title("Wind Speed vs LV Active Power (Actual vs Theoretical)",
                     color="white", fontsize=11, pad=10)
        ax.set_xlabel("Wind Speed (m/s)", color=GRAY, fontsize=10)
        ax.set_ylabel("Power (kW)",       color=GRAY, fontsize=10)
        ax.legend(facecolor=LGBG, labelcolor="white", fontsize=9)
        p = f"{OUTPUT_DIR}/plot4_scatter_power_curve.png"
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=DARK)
        plt.close(); saved.append(p); print(f"  Saved -> {p}")

        # PLOT 5: Outlier strip
        pwr_arr  = np.array(df[COL_POWER], dtype=np.float64)
        lo_f = s[COL_POWER]["lo_fence"]
        hi_f = s[COL_POWER]["hi_fence"]
        omask = (pwr_arr < lo_f) | (pwr_arr > hi_f)
        fig, ax = plt.subplots(figsize=(10, 4))
        _style(fig, ax)
        ax.scatter(np.where(~omask)[0], pwr_arr[~omask],
                   s=4, alpha=0.4, color=BLUE,
                   label=f"Normal ({(~omask).sum():,})")
        ax.scatter(np.where(omask)[0], pwr_arr[omask],
                   s=14, alpha=0.9, color=RED,
                   label=f"Outliers ({omask.sum():,})")
        ax.axhline(hi_f, color=ORANGE, ls="--", lw=1.5,
                   label=f"Upper fence = {hi_f:.1f} kW")
        ax.axhline(lo_f, color=ORANGE, ls=":", lw=1.2,
                   label=f"Lower fence = {lo_f:.1f} kW")
        ax.set_title("Outlier Detection - LV ActivePower (IQR Method)",
                     color="white", fontsize=11, pad=10)
        ax.set_xlabel("Record Index", color=GRAY, fontsize=10)
        ax.set_ylabel("LV Power (kW)", color=GRAY, fontsize=10)
        ax.legend(facecolor=LGBG, labelcolor="white", fontsize=9)
        p = f"{OUTPUT_DIR}/plot5_outlier_detection.png"
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=DARK)
        plt.close(); saved.append(p); print(f"  Saved -> {p}")

        # ANIMATION 1: Rolling mean power
        print("  Building animation 1 (rolling mean power)...")
        df_t = df.sort_values(COL_TIME).reset_index(drop=True)
        roll = (pd.Series(df_t[COL_POWER].values)
                .rolling(window=24, min_periods=1).mean().values)
        fig_a, ax_a = plt.subplots(figsize=(10, 4))
        _style(fig_a, ax_a)
        ax_a.set_xlim(0, len(df_t))
        ax_a.set_ylim(-50, df_t[COL_POWER].max() * 1.05)
        ax_a.set_title("Rolling Mean LV Power - January (0-36 deg Sector)",
                       color="white", fontsize=11, pad=10)
        ax_a.set_xlabel("Record (chronological)", color=GRAY, fontsize=10)
        ax_a.set_ylabel("LV Power (kW)", color=GRAY, fontsize=10)
        raw_l,  = ax_a.plot([], [], color=BLUE, alpha=0.3,
                             lw=0.7, label="Raw Power")
        roll_l, = ax_a.plot([], [], color=RED, lw=2.2,
                             label="Rolling Mean (w=24)")
        ax_a.legend(facecolor=LGBG, labelcolor="white", fontsize=9)
        step   = max(1, len(df_t) // 100)
        frames = list(range(24, len(df_t), step))

        def _upd1(frame):
            raw_l.set_data(range(frame), df_t[COL_POWER].values[:frame])
            roll_l.set_data(range(frame), roll[:frame])
            return raw_l, roll_l

        ani1 = animation.FuncAnimation(
            fig_a, _upd1, frames=frames, interval=40, blit=True)
        p = f"{OUTPUT_DIR}/anim1_rolling_mean_power.gif"
        ani1.save(p, writer="pillow", fps=18,
                  savefig_kwargs={"facecolor": DARK})
        plt.close(); saved.append(p); print(f"  Saved -> {p}")

        # ANIMATION 2: Power curve build-up
        print("  Building animation 2 (power curve evolution)...")
        ws_vals  = np.array(df[COL_WIND])
        pwr_vals = np.array(df[COL_POWER])
        sort_idx = np.argsort(ws_vals)
        ws_s     = ws_vals[sort_idx]
        pwr_s    = pwr_vals[sort_idx]
        fig_b, ax_b = plt.subplots(figsize=(8, 5))
        _style(fig_b, ax_b)
        ax_b.set_xlim(0, 26); ax_b.set_ylim(-100, 2200)
        ax_b.set_xlabel("Wind Speed (m/s)", color=GRAY, fontsize=10)
        ax_b.set_ylabel("LV Power (kW)", color=GRAY, fontsize=10)
        ws_line = np.linspace(0, 25, 300)

        def _tc(w):
            p = np.zeros_like(w)
            m1 = (w >= 3.5) & (w < 14)
            m2 = (w >= 14)  & (w <= 25)
            p[m1] = 2000 * ((w[m1]-3.5)/(14-3.5))**3
            p[m2] = 2000.0
            return np.clip(p, 0, 2000)

        ax_b.plot(ws_line, _tc(ws_line), color=ORANGE, lw=1.8,
                  ls="--", alpha=0.6, label="Theoretical", zorder=1)
        scat_b  = ax_b.scatter([], [], s=8, alpha=0.55,
                                color=BLUE, zorder=2, label="Actual")
        title_b = ax_b.set_title("", color="white", fontsize=11, pad=10)
        ax_b.legend(facecolor=LGBG, labelcolor="white", fontsize=9)
        n_frames  = 50
        step_b    = max(1, len(ws_s) // n_frames)
        frame_ends = list(range(step_b, len(ws_s), step_b))
        if not frame_ends or frame_ends[-1] < len(ws_s):
            frame_ends.append(len(ws_s))

        def _upd2(i):
            end = frame_ends[i]
            scat_b.set_offsets(np.c_[ws_s[:end], pwr_s[:end]])
            title_b.set_text(f"Power Curve Build-Up | "
                             f"Wind <= {ws_s[end-1]:.2f} m/s ({end:,} pts)")
            return scat_b, title_b

        ani2 = animation.FuncAnimation(
            fig_b, _upd2, frames=len(frame_ends), interval=300, blit=False)
        p = f"{OUTPUT_DIR}/anim2_power_curve_buildup.gif"
        ani2.save(p, writer="pillow", fps=4,
                  savefig_kwargs={"facecolor": DARK})
        plt.close(); saved.append(p); print(f"  Saved -> {p}")

        print(f"\n  All {len(saved)} outputs saved to '{OUTPUT_DIR}/'")
        return saved

    def run(self):
        self.ingest()
        self.clean()
        self.analyze()
        self.visualize()
        print("\n" + "=" * 62)
        print("  Pipeline complete. All outputs ready.")
        print("=" * 62)


if __name__ == "__main__":
    pipeline = WindTurbinePipeline(DATA_PATH)
    pipeline.run()
