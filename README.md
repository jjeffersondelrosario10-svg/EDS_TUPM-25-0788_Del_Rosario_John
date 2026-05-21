# EDS_TUPM-25-0788_DelRosario

**Engineering Data Systems Pipeline — RMS-01**
**Wind Turbine SCADA Analytics**

**Author:** John Jefferson DA. Del Rosario
**Student ID:** TUPM-25-0788 | BS Mechanical Engineering
**School:** Technological University of the Philippines, Manila
**Course:** Computer Programming | AY 2026

---

## Dataset
Real Kaggle dataset: **berkerisen/wind-turbine-scada-dataset**
Download T1.csv from: https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset
Place it in the `data/` folder as `T1.csv`

## Unique Filter (Birthday: January 10, 2007)
| Filter | Value | Derivation |
|--------|-------|-----------|
| Month | January (1) | Birth month |
| Wind Direction | 0° – 36° | Birth day 10 × 3.6°/day |
| Random seed | 2007 | Birth year |

## Repository Structure
```
EDS_TUPM-25-0788_DelRosario/
├── main.py                  
├── requirements.txt
├── README.md
├── datasets/
│   ├── T1.csv               
│   └── dataset_cleaned.csv  
└── outputs/
    ├── plot1_power_histogram_kde.png
    ├── plot2_boxplot_wind_groups.png
    ├── plot3_correlation_heatmap.png
    ├── plot4_scatter_power_curve.png
    ├── plot5_outlier_detection.png
    ├── anim1_rolling_mean_power.gif
    └── anim2_power_curve_buildup.gif
```

## How to Run
```bash
pip install -r requirements.txt
python main.py
```


