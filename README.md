# WF_GoNogo
A pipeline for analyzing widefield calcium imaging data from mice performing a interhemispheric Go/No-go visual task. This repository includes analysis of behavioral performance, pupil dynamics, locomotion speed, widefield imaging data, and licking behavior.
## Project Structure
```plaintext
WF_GoNogo/
├── mouse_data/               # Symbolic links or pointers to organized raw data (do not store raw here)
│   ├── Mouse01/
│   │   ├── behavior/         # Behavioral CSVs 
│   │   ├── pupil/            # Pupil videos 
│   │   └── wf/               # Widefield imaging data
│   ├── Mouse02/
│   │   └── ...
│   └── ...
│
├── analysis/                 # Main analysis scripts
│   ├── run_behavior.ipynb
│   ├── run_pupil.ipynb
│   ├── run_wf.ipynb
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
- pupil/ – pupil video 
- wf/ – widefield imaging files
2. Configure settings
Create a YAML file under config/ for each mouse/session. A typical file might define:
```yaml
mouse_id: A092
birth_date: '2025-05-12'
base_dir: /home/lsh/Data_attention/Transfer learning/DATA_linshu
sessions:               
  - '20250729'
  - '20250730'
  - '20250731'
  - '20250801'
```
3. Run pipeline
