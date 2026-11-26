# Multi-Model Banking FTI + MLOps Level 2 Platform

Production-oriented banking ML platform combining the [**Hopsworks FTI (Feature / Training / Inference)**](https://www.hopsworks.ai/post/mlops-to-ml-systems-with-fti-pipelines) system architecture with [**Google Cloud MLOps Level 2 (CI / CD / CT)**](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) automation on a **$0.00/month** open-source and free-tier stack.

## Architecture Overview

- **Multi-Model Cross-Feature Reuse:** Both **Batch Credit Risk (`credit_risk_batch` — LightGBM -> ONNX)** and **Real-Time Fraud Detection (`fraud_online` — XGBoost -> ONNX)** consume shared customer profile and strictly causal `t - 1` rolling card-window aggregations from a unified **Feast Feature Store**.
- **Strict FTI Decoupling & Unified Serving:** Training pipelines read point-in-time joins solely from the Feature Store and export models alongside train-split-only preprocessors to **ONNX Runtime** (`<150 MB` serving RSS, zero training libraries in Feature or Inference runtimes).
- **Zero-Cost Cloud & Local Reproducibility:** Engineered around DVC + DagsHub MLflow 3, Feast (Offline Parquet + Neon Serverless Postgres Online Store), Google Colab CLI GPU Continuous Training, Render Free Tier, and GitHub Pages — with 1-command local Docker Compose reproduction requiring zero cloud credentials.

## PoC-First 4-Stage Roadmap

| Stage | Scope | Status |
| :--- | :--- | :--- |
| **Day-0** | Banking GitFlow CI, Conventional Branch/Commits, SemVer & Auto-PR Chain (`00`) | Implemented |
| **Stage 1** | Local PoC — `credit_risk_batch` (`F -> T -> I-Batch`, Feast Offline, MLflow, Docker) | Planned |
| **Stage 2** | Cloud PoC — `credit_risk_batch` (`dev/hom/prd`, Colab GPU CT, GitHub Pages, Rollback) | Planned |
| **Stage 3** | Local Expansion — `fraud_online` (Feast Online Store, FastAPI ONNX API, Dual Stack) | Planned |
| **Stage 4** | Cloud `fraud_online` (Neon `hom`/`prd`, Render SLA), Dual Evidently Drift & Showcase | Planned |

## Banking GitFlow & Contributor Rules

All contributors and automated agents follow strict English **Gitflow**, **Conventional Branch 1.0.0**, **Conventional Commits 1.0.0**, and **Semantic Versioning 2.0.0**:

- **Local development:** Exclusively on `feature/<kebab-case-description>` branches cut from `develop`.
- **Commit headers & PR titles:** `<type>[optional scope][!]: <description>` in lowercase imperative English (`<= 72` chars, no trailing period).
- **Automated PR promotion (manual merge mandatory):**
  1. `push` to `feature/*` -> `ci.yml` -> auto-opens PR `feature/* -> develop` (manual merge).
  2. Merge to `develop` -> `cd-pipeline.yml` (`dev`) -> auto-opens PR `develop -> release/vX.Y.Z` (manual merge).
  3. Merge to `release/vX.Y.Z` -> `cd-pipeline.yml` (`hom`) -> auto-opens PR `release/vX.Y.Z -> main` (manual merge).

### GitHub Repository Settings & Branch Protection

- **Actions PR permissions:** Enable **Allow GitHub Actions to create and approve pull requests** under **Settings -> Actions -> General -> Workflow permissions**.
- **No auto-merge:** Disable **Allow auto-merge** under **Settings -> General -> Pull Requests** so every promotion PR requires manual human review and merge.
- **Branch protections / rulesets:**
  - `develop`: require PRs, require `Governance, Lint, Typecheck & Tests` (`ci.yml`), block direct/force pushes.
  - `release/v*`: require PRs (`develop -> release/vX.Y.Z`), require `Dev Validation (develop -> dev)`, grant `GITHUB_TOKEN` / automation ruleset bypass to create `release/vX.Y.Z` from `main`, and delete merged release branches.
  - `main`: require PRs (`release/vX.Y.Z -> main`), require `Homologation Validation (release/v* -> hom)`, block direct/force pushes, and configure GitHub Environments (`dev`, `hom`, `prd` with required reviewers on `prd` in Stage 2+).

## Local Development

```bash
make install   # Install package + dev dependencies with uv
make check     # Run ruff lint, ruff format check, strict mypy, and offline pytest
```

## Architectural References

- **Hopsworks FTI Architecture:** [From MLOps to ML Systems with Feature/Training/Inference Pipelines](https://www.hopsworks.ai/post/mlops-to-ml-systems-with-fti-pipelines)
- **Google Cloud MLOps Level 2:** [MLOps: Continuous delivery and automation pipelines in machine learning](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
