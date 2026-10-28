# Cheminformatics Virtual Laboratory

A Streamlit-based undergraduate chemistry and cheminformatics learning platform built on the supplied SRM-branded project. It keeps the original molecular visualization, descriptor analysis, structure–property comparison, assessments, and PDF report workflows while adding experiment simulations, an optional AI tutor, and persistent local progress storage.

## Product modules

### Molecular science
- SMILES input and example molecule picker
- 2D molecular structure depiction and generated 3D conformers
- 3D rendering with py3Dmol
- Molecular formula, molecular weight, exact mass, LogP, TPSA, H-bond donors/acceptors, rotatable bonds, rings, atom/bond counts, and other descriptors
- Structure–property comparison and descriptor dashboard
- Functional group explorer, atom/bond explorer, and curated molecule library
- Downloadable CSV outputs and PDF molecular/final-project reports

### Virtual experiments
- Strong acid–strong base titration with an idealized pH curve, exact selected-volume recalculation, and equivalence-volume estimate
- Improved titration pH model solving the acid/base charge balance with water autoionization and a temperature-dependent pKw approximation (0–100 °C)
- More consistent chart styling, cleaner navigation presentation, and responsive UI styling
- First-order reaction kinetics, concentration-vs-time plot, linearized plot, and half-life
- Nernst equation calculator
- Beer–Lambert law calculator and calibration line
- Arrhenius equation temperature/rate-constant simulation
- CSV exports and explicit “record experiment” actions

### Learning and assessment
- Built-in offline chemistry tutor guide
- Optional live AI tutor using an OpenAI-compatible Chat Completions API
- Practice quiz with answer explanations and score history
- Activity history, persistent quiz history, summary metrics, CSV/JSON exports
- SQLite database for local persistence; registration numbers are hashed to create pseudonymous student keys and are not written to the database

## Run locally

Recommended: Python 3.11. RDKit is a compiled dependency; if your Python version/platform cannot install a compatible wheel, use Docker.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Optional live AI tutor

The app works without an AI key using the curated offline tutor. To enable generated, question-specific answers, configure a provider that supports the OpenAI-compatible `/chat/completions` endpoint.

### Local setup
1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
2. Add your provider's API key, base URL, and model name.
3. Restart Streamlit.

Never commit `.streamlit/secrets.toml`. `.gitignore` excludes it. The AI feature sends the student's question and the selected topic to the configured provider; do not submit personal or confidential information. API usage may incur provider charges.

### Streamlit Community Cloud
Open the app's **Settings → Secrets** and add:

```toml
CHEM_LAB_AI_API_KEY = "your-provider-api-key"
CHEM_LAB_AI_BASE_URL = "https://api.openai.com/v1"
CHEM_LAB_AI_MODEL = "gpt-4o-mini"
```

Use the endpoint and model actually supported by your chosen provider. Do not paste secrets into the app's text fields or commit them to GitHub.

## Persistence and student identity

- Local execution stores data in `data/chem_lab.db` by default.
- Override the location with `CHEM_LAB_DB_PATH`.
- The raw registration number is not stored; it is hashed to associate activity and quiz records. If a registration number is not supplied, the record is scoped to the browser session to avoid merging students who share a name.
- SQLite is suitable for a local/single-instance educational deployment. Streamlit Community Cloud's local filesystem should not be treated as durable storage across rebuilds or as a multi-instance database.
- For real multi-student production use, add institutional authentication and a managed database (for example, PostgreSQL/Supabase) with authorization rules, backups, retention policy, and appropriate privacy notices. This project does not claim to provide secure account authentication.

## Docker deployment

Docker keeps the SQLite database in a named volume, so ordinary container recreation does not discard it.

```bash
docker compose up --build
```

Open `http://localhost:8501`. To enable AI, uncomment the provider environment variables in `docker-compose.yml` or configure them in your deployment environment. For public production hosting, use a secret manager rather than committing credentials.

## Run tests

```bash
pip install -r requirements-dev.txt
python -m py_compile app.py storage.py ai_tutor.py chemistry_core.py
python -m unittest discover -s tests -v
```

The automated tests cover the core calculation functions and SQLite persistence. They do not replace browser-level UI testing or a deployment smoke test.

## Deploy to Streamlit Community Cloud

1. Extract the ZIP and push the project files to a GitHub repository you control.
2. Keep `app.py`, `storage.py`, `ai_tutor.py`, `chemistry_core.py`, `requirements.txt`, `packages.txt`, `srm_logo.jpeg`, and the rest of the project files together.
3. In Streamlit Community Cloud, connect the repository and select `app.py` as the main file.
4. Configure AI secrets only if you want live AI tutor responses.
5. After the build finishes, test molecular depiction, 3D rendering, PDF downloads, each experiment, quiz submission, and progress export.

Updating this ZIP does **not** update the already-deployed Streamlit URL by itself; the connected repository must be updated and redeployed.

## Scientific and safety limitations

- Titration is an idealized strong monoprotic acid/strong base model. It solves the charge balance with water autoionization and an approximate temperature-dependent pKw (0–100 °C), but does not include activity coefficients, ion pairing, or detailed equilibrium chemistry. At high concentrations, pH based on concentration is not the same as activity-based pH.
- First-order kinetics assumes a single first-order process at constant temperature.
- The Nernst calculator requires a user-supplied reaction quotient and electron count; it does not balance redox reactions automatically.
- Beer–Lambert assumes a linear absorbance response and a single absorbing species.
- Arrhenius calculations illustrate a model and depend on user-supplied parameters.
- Functional-group pattern matches are educational hints and may overlap; chemical context matters.
- Generated 3D conformers are computed geometries, not necessarily experimentally observed or global-minimum structures.
- AI explanations can be incorrect; verify course-specific answers against lecturer notes and trusted references.
- These simulations are for education and are not a substitute for real laboratory supervision, institutional safety rules, or professional chemical analysis.

## Main files

- `app.py` — Streamlit UI and original cheminformatics workflows
- `chemistry_core.py` — validated scientific calculations
- `storage.py` — SQLite persistence and pseudonymous student keys
- `ai_tutor.py` — optional AI-provider integration with offline fallback handled by the UI
- `tests/` — unit tests for science calculations and persistence
- `Dockerfile`, `docker-compose.yml` — containerized local deployment
