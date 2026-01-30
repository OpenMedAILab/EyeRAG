# EyeRAG GitHub Release - Excluded Files and Directories

The following files and directories are excluded from the GitHub release for security, privacy, and size considerations:

## Sensitive Information
- `.env` - Contains actual API keys and credentials
- `api_keys.py` - Contains API keys
- `*.pem` - Private key files
- `*.key` - Key files
- `credentials.json` - Credential files
- `secrets.json` - Secret files

## Large Data Files
- `Data/` - Entire data directory (medical guides, patient data, etc.)
- `RESULTS/` - Results directory
- `results/` - Results directory
- `Data/RAG/AAO_PPP/` - AAO PPP medical guide data
- `Data/RAG/COS/back/` - Backup medical guide data
- `Data/RAG/COS/OphthaKG_COS/` - Ophthalmology knowledge graph data
- `Data/RAG/COS/faiss_indexes/` - FAISS index files
- `Data/RAG/COS/medical_guide_md_data.pkl` - Pickle file with medical guide data
- `Data/RAG/COS/medical_guide_markdown/` - Full medical guide markdown files (excluded except for essential subset)
- `Data/RAG/COS/medical_guide_markdown_summary/` - Medical guide summary files
- `Data/patient_data.xlsx` - Patient data spreadsheet
- `*.zip` - Zip archives
- `*.tar.gz` - Compressed archives

## Temporary and Build Files
- `tmp/` - Temporary files
- `back/` - Backup directory
- `to_del/` - Files marked for deletion
- `__pycache__/` - Python cache directories
- `*.py[cod]` - Python compiled files
- `*$py.class` - Python class files
- `*.so` - Shared object files
- `*.egg-info/` - Package info
- `.Python` - Python virtual environment
- `build/` - Build directory
- `develop-eggs/` - Development eggs
- `dist/` - Distribution directory
- `downloads/` - Downloads
- `eggs/` - Eggs directory
- `.eggs/` - Hidden eggs directory
- `lib/` - Library directory
- `lib64/` - 64-bit library directory
- `parts/` - Parts directory
- `sdist/` - Source distribution
- `var/` - Variable files
- `wheels/` - Wheel packages
- `.installed.cfg` - Installation config
- `*.egg` - Egg packages

## IDE and Editor Files
- `.idea/` - IntelliJ/PyCharm IDE files
- `.vscode/` - VSCode settings
- `*.swp` - Vim swap files
- `*.swo` - Vim swap files
- `*~` - Emacs backup files
- `.project` - Project files
- `.pydevproject` - PyDev project files
- `.settings/` - IDE settings
- `*.sublime-project` - Sublime project files
- `*.sublime-workspace` - Sublime workspace files

## Jupyter Notebook Files
- `.ipynb_checkpoints/` - Notebook checkpoints
- `*.ipynb` - Jupyter notebooks

## Test and Coverage Files
- `.pytest_cache/` - Pytest cache
- `.coverage` - Coverage reports
- `htmlcov/` - HTML coverage reports
- `.tox/` - Tox files
- `.nox/` - Nox files

## Virtual Environments
- `venv/` - Virtual environment
- `env/` - Environment directory
- `ENV/` - Environment directory
- `.venv/` - Hidden virtual environment
- `test_env/` - Test environment

## Log and Temporary Files
- `*.log` - Log files
- `*.tmp` - Temporary files
- `*.temp` - Temporary files
- `*.bak` - Backup files
- `logs/` - Logs directory

## OS Generated Files
- `.DS_Store` - macOS system files
- `__MACOSX/` - macOS archive files
- `._*` - macOS resource forks

## Claude Code Files
- `.claude/` - Claude code files

## LightRAG Generated Files
- `lightrag_build.log` - LightRAG build log
- `processed_files.txt` - Processed files list

## Development/Personal Files
- `eye_rag/tools/eval_eye_rag_extended.py` - Extended evaluation tool
- `paper/` - Paper-related files and experiments
- `tests/` - Test files
- `simhei.ttf` - Font file (included in release)

## Results and Experiment Data
- `RESULTS/` - Main results directory
- `results/` - Results directory
- `paper/results/` - Paper results
- `paper/experiments/` - Experiment files
- `paper/entity_relationship/` - Entity relationship data
- `paper/check_entity_relationship.py` - Entity relationship checker
- `paper/count_num_tokens.py` - Token counting script
- `paper/eval_eye_rag_extended.py` - Extended evaluation script
- `paper/image_to_tiff_converter.py` - Image converter
- `paper/plot_result.py` - Result plotting script
- `paper/test_converter.py` - Test converter
- `paper/translator.py` - Translator script
- `paper/example_tiff_conversion.py` - TIFF conversion example
- `paper/plot_result.py` - Plotting results
- `paper/check_entity_relationship.py` - Entity relationship checker
- `paper/count_num_tokens-v0.py` - Token counting script
- `paper/count_num_tokens.py` - Token counting script

## Deprecated Code
- `lightrag/deprecated/` - Deprecated LightRAG code
- `eye_rag/deprecated/` - Deprecated EyeRAG code

## Pinecone Integration
- `eye_rag/rag/pinecone/` - Pinecone integration files

## Backups and Temp Directories
- `back/` - Backup directory
- `to_del/` - Files to delete
- `tmp/` - Temporary files
- `test_env/` - Test environment directory

This exclusion list ensures that only essential code, documentation, and configuration files are included in the public GitHub release, maintaining security and keeping the repository size manageable.