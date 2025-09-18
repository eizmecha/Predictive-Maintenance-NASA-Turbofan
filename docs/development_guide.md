# Development Guide & Workflow

## Team Roles & Responsibilities
| Member   | Role             | Responsibilities |
|----------|------------------|------------------|
| Eiz      | Project Lead     | Coordination, optimization, documentation |
| Osamah   | Data Engineer    | Preprocessing, model development, code structure |
| Ali      | Data Analyst     | EDA, visualization, results interpretation |

---

## Project Phases
1. **Setup** → repo, environment, structure.  
2. **EDA** → analyzed dataset, identified critical sensors.  
3. **Preprocessing** → built feature engineering & normalization pipeline.  
4. **Baseline Models** → LR, RF, XGBoost for benchmarking.  
5. **Optimization** → hyperparameter tuning, LSTM implementation.  
6. **Finalization** → test evaluation, demo, documentation.  

---

## Contribution Workflow
1. Create feature branch → `git checkout -b feature/new-feature`  
2. Implement changes (use `src/utils/` helpers to keep DRY).  
3. Test with notebooks or `main.py`.  
4. Commit changes → `git commit -m "feat: added new feature"`  
5. Push to branch → `git push origin feature/new-feature`  
6. Open Pull Request → peer review before merging.  
