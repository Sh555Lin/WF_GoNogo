# WF_GoNogo
A pipeline for analyzing widefield calcium imaging data from mice performing a interhemispheric Go/No-go visual task. This repository includes analysis of behavioral performance, pupil dynamics, locomotion speed, widefield imaging data, and licking behavior.
## Project Structure
```plaintext
WF_GoNogo/
├── mouse_data/               # Symbolic links or pointers to organized raw data (do not store raw here)
│   ├── Mouse01/
│   │   ├── behavior/         # Behavioral CSVs or raw logs
│   │   ├── pupil/            # Pupil videos or tracking results
│   │   └── wf/               # Widefield imaging data
│   ├── Mouse02/
│   │   └── ...
│   └── ...
│
├── analysis/                 # Main analysis scripts
│   ├── run_behavior.py
│   ├── run_pupil.py
│   ├── run_wf.py
│   └── combine_results.py    # Optional: merge multi-modal outputs
│
├── utils/                    # Helper functions and shared code
│   ├── io.py
│   ├── plotting.py
│   └── preprocessing.py
│
├── config/                   # Config files for each batch/mouse
│   ├── Mouse01_config.yaml
│   ├── Mouse02_config.yaml
│   └── ...
│
├── results/                  # Output results organized by mouse and session
│   ├── Mouse01/
│   ├── Mouse02/
│   └── ...
│
├── figures/                  # Final figures for publications or QC
│   └── ...
│
├── run_all.py                # One-click pipeline execution script
├── requirements.txt          # Python dependencies
└── README.md                 # Project description and usage
```
## Dependencies
Install required Python packages:

```bash
pip install -r requirements.txt
```
## Usage
1. Organize data
Ensure your data follows this symbolic structure under mouse_data/. Each mouse folder should include:
- behavior/ – behavioral results in CSV format
- pupil/ – pupil video or tracking results (e.g., DeepLabCut outputs)
- wf/ – widefield imaging files (e.g., TIFFs, NumPy arrays, etc.)
2. Configure settings
Create a YAML file under config/ for each mouse/session. A typical file might define:
```yaml
mouse_id: Mouse01
session_id: 2025-08-01
frame_rate: 20
stim_timestamps_csv: behavior/stimulus_times.csv
wf_data_path: wf/session1.npy
pupil_video_path: pupil/pupil_cam1.avi
output_dir: results/Mouse01/session1/
```
3. Run pipeline
Run each modality step-by-step or use the wrapper:
```bash
python run_all.py --config config/Mouse01_config.yaml
```
Or run each module:

```bash
python analysis/run_behavior.py --config config/Mouse01_config.yaml
python analysis/run_pupil.py --config config/Mouse01_config.yaml
python analysis/run_wf.py --config config/Mouse01_config.yaml
```
4. View results
Processed data and intermediate outputs will be saved in the results/ directory. Final figures can be saved in figures/.
