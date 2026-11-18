PathSnap - member3_ml: LightGBM model, Optuna tuning, decoder re-tuning, final analysis, integrity checks

1. In your clone of the PathSnap repo:  git pull
2. Unzip THIS zip into the repo root (the folder that contains .git),
   overwriting existing files. Do not unzip into a sub-folder.
3. Commit and push from your own account:
     git add "AI_DISCLOSURE.md" "artifacts/decoder_config.json" "artifacts/model.joblib" "artifacts/model_meta.json" "experiments/03_train_compare.py" "experiments/04_hmm_tune_compare.py" "experiments/05_optuna_tune.py" "experiments/06_final_analysis.py" "experiments/07_integrity_checks.py" "requirements.txt" "results/experiments_log.csv" "results/phase2/cv_folds.csv" "results/phase2/fig_confusion_matrices.png" "results/phase2/fig_cost.png" "results/phase2/fig_cv_macro_f1.png" "results/phase2/fig_log_loss.png" "results/phase2/fig_test_top1_macro_f1.png" "results/phase2/model_comparison.csv" "results/phase2/per_trip_top1.csv" "results/phase2/train_run_log_lightgbm.txt" "results/phase5/decoder/comparison_summary.csv" "results/phase5/decoder/decoder_grid.csv" "results/phase5/decoder/fig_decoder_grid.png" "results/phase5/decoder/fig_hmm_vs_classifier.png" "results/phase5/decoder/per_trip_comparison.csv" "results/phase5/decoder_final/comparison_summary.csv" "results/phase5/decoder_final/decoder_config.json" "results/phase5/decoder_final/decoder_grid.csv" "results/phase5/decoder_final/fig_decoder_grid.png" "results/phase5/decoder_final/fig_hmm_vs_classifier.png" "results/phase5/decoder_final/per_trip_comparison.csv" "results/phase5/decoder_final_run_log.txt" "results/phase5/decoder_lightgbm/comparison_summary.csv" "results/phase5/decoder_lightgbm/decoder_config.json" "results/phase5/decoder_lightgbm/decoder_grid.csv" "results/phase5/decoder_lightgbm/fig_decoder_grid.png" "results/phase5/decoder_lightgbm/fig_hmm_vs_classifier.png" "results/phase5/decoder_lightgbm/per_trip_comparison.csv" "results/phase5/decoder_lightgbm_run_log.txt" "results/phase5/decoder_run_log.txt" "results/phase5/error_categories.csv" "results/phase5/fig_confusion_tuned_vs_untuned.png" "results/phase5/fig_error_categories.png" "results/phase5/fig_final_per_trip.png" "results/phase5/fig_optuna_history.png" "results/phase5/fig_optuna_params.png" "results/phase5/fig_tuned_vs_untuned.png" "results/phase5/final_analysis_log.txt" "results/phase5/final_model_selection.csv" "results/phase5/final_per_trip.csv" "results/phase5/final_summary.csv" "results/phase5/integrity_checks.txt" "results/phase5/map_worst_00000007.html" "results/phase5/map_worst_00000009.html" "results/phase5/map_worst_00000077.html" "results/phase5/optuna_run_log.txt" "results/phase5/optuna_trials.csv" "results/phase5/tuned_vs_untuned.csv"
     git commit -m "Add LightGBM comparison, Optuna tuning, final analysis and integrity checks"
     git push
   (if push is rejected: git pull --rebase, then git push)

Files (58):
  AI_DISCLOSURE.md
  artifacts/decoder_config.json
  artifacts/model.joblib
  artifacts/model_meta.json
  experiments/03_train_compare.py
  experiments/04_hmm_tune_compare.py
  experiments/05_optuna_tune.py
  experiments/06_final_analysis.py
  experiments/07_integrity_checks.py
  requirements.txt
  results/experiments_log.csv
  results/phase2/cv_folds.csv
  results/phase2/fig_confusion_matrices.png
  results/phase2/fig_cost.png
  results/phase2/fig_cv_macro_f1.png
  results/phase2/fig_log_loss.png
  results/phase2/fig_test_top1_macro_f1.png
  results/phase2/model_comparison.csv
  results/phase2/per_trip_top1.csv
  results/phase2/train_run_log_lightgbm.txt
  results/phase5/decoder/comparison_summary.csv
  results/phase5/decoder/decoder_grid.csv
  results/phase5/decoder/fig_decoder_grid.png
  results/phase5/decoder/fig_hmm_vs_classifier.png
  results/phase5/decoder/per_trip_comparison.csv
  results/phase5/decoder_final/comparison_summary.csv
  results/phase5/decoder_final/decoder_config.json
  results/phase5/decoder_final/decoder_grid.csv
  results/phase5/decoder_final/fig_decoder_grid.png
  results/phase5/decoder_final/fig_hmm_vs_classifier.png
  results/phase5/decoder_final/per_trip_comparison.csv
  results/phase5/decoder_final_run_log.txt
  results/phase5/decoder_lightgbm/comparison_summary.csv
  results/phase5/decoder_lightgbm/decoder_config.json
  results/phase5/decoder_lightgbm/decoder_grid.csv
  results/phase5/decoder_lightgbm/fig_decoder_grid.png
  results/phase5/decoder_lightgbm/fig_hmm_vs_classifier.png
  results/phase5/decoder_lightgbm/per_trip_comparison.csv
  results/phase5/decoder_lightgbm_run_log.txt
  results/phase5/decoder_run_log.txt
  results/phase5/error_categories.csv
  results/phase5/fig_confusion_tuned_vs_untuned.png
  results/phase5/fig_error_categories.png
  results/phase5/fig_final_per_trip.png
  results/phase5/fig_optuna_history.png
  results/phase5/fig_optuna_params.png
  results/phase5/fig_tuned_vs_untuned.png
  results/phase5/final_analysis_log.txt
  results/phase5/final_model_selection.csv
  results/phase5/final_per_trip.csv
  results/phase5/final_summary.csv
  results/phase5/integrity_checks.txt
  results/phase5/map_worst_00000007.html
  results/phase5/map_worst_00000009.html
  results/phase5/map_worst_00000077.html
  results/phase5/optuna_run_log.txt
  results/phase5/optuna_trials.csv
  results/phase5/tuned_vs_untuned.csv