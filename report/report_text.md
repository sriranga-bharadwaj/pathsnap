# PathSnap report text (DRAFT - the team must rewrite this in its own words)
# Format: each block starts with "@@ key". Blank line = new paragraph. Lines starting "- " = bullets.
# "### Heading" starts a small heading. **bold**, `code`. {name} = value filled from results/ at build time.
# Rebuild after editing:  .venv\Scripts\python.exe report/build_report.py
# Keep to plain ASCII characters (no arrows / >= / smart quotes): the PDF fonts may not have them.

@@ title_team
**Team:**

- SRIJAN AKSHIT - PES1UG24AM284
- SRIPADA KRISHNA SAMEER ADITYA - PES1UG24AM285
- SRIRANGA BHARADWAJ - PES1UG24AM286
- SRIVAMSHI CHIKKIREDDI - PES1UG24AM287

**GitHub repository:** https://github.com/sriranga-bharadwaj/pathsnap

@@ title_summary
PathSnap matches noisy GPS trajectories to a road network. A classifier scores every (GPS observation, nearby road arc) pair and an HMM/Viterbi decoder turns these scores into one connected route. The deployed system (untuned Random Forest + HMM) reaches an observation-level Top-1 accuracy of {hmm_top1_pooled} on the 10 student-test trips (classifier alone: {clf_top1_pooled}), with route continuity {hmm_route_continuity_trip_macro} and route-level arc-set F1 {hmm_arc_f1_trip_macro}. It runs as a FastAPI service on Laptop 1 and a static web UI on Laptop 2.

@@ overview
The task is framed as candidate-based map matching. For every GPS fix we collect the road arcs (directed road segments) close to it, a machine-learning classifier estimates how likely each arc is the one the vehicle was on, and an HMM/Viterbi decoder picks the sequence of arcs that is both likely under the classifier and drivable on the network. The table below shows the stages; all of them run on Laptop 1 except the UI.

@@ eda_intro
The data is organised by trip. Each trip folder contains `.track` (longitude, latitude, timestamp in seconds), `.nodes` (longitude, latitude; row index = node id), `.arcs` (from node, to node; row index = arc id) and `.route` (the reference arc sequence). Every trip has its **own** road network, so all network processing is done per trip. All analyses below use the **train** partition ({tr_trips} trips) and the **student_test** partition ({st_trips} trips); the partition is stated in every table and figure caption. The evaluator partition was never available or used. Script: `experiments/00_eda.py`.

@@ eda_preprocess
- **Parsing:** whitespace-separated files without headers; `.track` files use CRLF line endings. Every row was checked for the expected number of columns.
- **Projection:** trips are spread over several countries, so no single UTM zone fits. Each trip is projected with a local azimuthal-equidistant (AEQD) projection centred on its median GPS position. The farthest point is {max_dist_centre_km} km from its centre and the largest difference between projected and geodesic step lengths is {proj_err_pct}, so distances in metres are reliable.
- **Track features:** time step, speed, acceleration and movement heading (from the neighbouring fixes). A time gap above 60 s starts a new segment, where the decoder restarts.
- **Network:** each arc becomes a straight segment between its two nodes, with its length, direction and reverse twin (the arc in the opposite direction, if any). Arcs are indexed with an STR-tree for fast radius queries.
- **Labels (training and evaluation only):** `.route` is an arc sequence, not one label per fix, so every fix was aligned to the route with a monotonic dynamic programme (distance + heading penalty, skipping arcs allowed). Each label gets a confidence flag (high / medium / low) that is used as a sample weight (1.0 / 0.5 / 0.2).
- **Trip-quality gate (train only):** trips where more than 10% of fixes end up over 50 m from their own reference route are excluded from training.

@@ eda_missing
Across both partitions the total count of missing values, invalid coordinates, duplicate rows, duplicate timestamps and non-increasing timestamps is {missing_total}, and the number of isolated network nodes is {isolated_total}. The data is therefore complete; the quality issues are of a different kind: irregular sampling, long gaps, and reference routes that do not always cover the GPS track (Table 1).

@@ eda_patterns
- **Size varies a lot:** trips have {all_points_min} to {all_points_max} GPS observations and networks have {all_arcs_min} to {all_arcs_max} arcs (Fig 2, Fig 5), so the spatial index is built per trip and cached.
- **Sampling:** nominally 1 Hz ({tr_pooled_share_dt_1s} of train steps and {st_pooled_share_dt_1s} of student-test steps are exactly 1 s), but the per-trip share drops to {tr_share_dt_1s_min}; the largest train gap is {tr_dt_max} s and there are {tr_gaps_gt_60s} gaps above 60 s in train ({st_gaps_gt_60s} in student_test). {n_frac_t_trips} train trips ({frac_t_trips}) have fractional timestamps (Fig 3).
- **Two directions per road:** the median share of arcs with a reverse twin is {twin_median} (lowest {twin_min}, trip {twin_min_trip}). Distance alone cannot tell the direction, so heading is essential.
- **GPS noise is small on student_test:** median distance to the reference route {st_route_p50} m, 99th percentile {st_route_p99} m, {st_route_gt50} of fixes beyond 50 m (Fig 6).
- **Reference-route limitations:** {route_breaks_total} places across {trips_with_breaks_total} trips where consecutive route arcs are not connected; {n_multi_scc_routes} routes cross strongly-connected components of the directed network; some train routes cover only part of the GPS track (Fig 7). Trip 00000054 is mostly off the supplied network (median {t54_net_p50} m to the nearest arc). We treat `.route` as reference information, not perfect ground truth.
- **Sparse sampling relative to arcs:** median {tr_gps_per_route_arc_median} fixes per route arc in train, {n_gps_per_arc_lt2} trips below 2, so consecutive fixes may skip whole arcs; the decoder must allow multi-arc transitions.

@@ eda_labels
After alignment, {al_tr_high} of the {al_tr_points} fixes in the {gate_kept} gate-kept train trips are high confidence, {al_tr_medium} medium and {al_tr_low} low (student_test: {al_st_high} / {al_st_medium} / {al_st_low}). The gate excluded {gate_excluded} (share of fixes beyond 50 m); {gate_warn} were kept with a warning. The gate was decided on train data only; student_test is never gated.

Candidates are all arcs within a fixed radius (both directions of a road are kept together), capped at 24 per fix, with one retry at 100 m when nothing is found. We swept 15 / 30 / 50 m (Table 2). 30 m is the smallest radius whose train recall ({rec_tr_30}) is within 0.5 points of the 50 m recall ({rec_tr_50}) while needing far fewer candidates ({cand_tr_30} vs {cand_tr_50} per fix). The cap costs almost nothing (uncapped recall {rec_tr_30_uncapped}) but bounds the worst case (up to {cand_max_uncapped_30} candidates uncapped). On student_test the candidate miss rate at 30 m is {miss_st_30}.

@@ eda_features
Each (observation, candidate arc) row has 9 features. All are location-independent (no coordinates, arc ids or trip ids), because the evaluator trips may be in unseen cities:

- `dist` - perpendicular distance from the fix to the arc: the main signal, GPS noise is a few metres.
- `heading_cos`, `heading_valid` - cosine of the angle between movement direction and arc direction, plus a flag when either is undefined: separates the two directions of the same road.
- `frac` - where the fix projects along the arc (0 = start, 1 = end): helps near junctions.
- `rank`, `rel_dist` - rank by distance and distance minus the nearest candidate's distance: make the score relative to the competing candidates of the same fix.
- `speed`, `accel`, `dt` - motion context: at low speed the heading is unreliable, and large time gaps weaken the evidence.

No window or neighbourhood-support features were used; sequence information is left to the HMM. Preprocessing (standard scaling where needed) is fitted on training rows only, inside each sklearn Pipeline.

@@ models_intro
Each model is a binary classifier for "is this candidate the true arc of this fix?". Training uses the 77 gate-kept train trips with at most 800 fixes per trip (seeded subsample: {n_train_obs} fixes, {n_train_rows} candidate rows); fixes whose reference arc is not among their candidates are left out of training but counted as errors in every metric. Cross-validation is GroupKFold(5) grouped by trip, so a trip is never in both the training and the validation fold. The 10 student-test trips are used only for comparison. Script: `experiments/03_train_compare.py`. Six models were trained: the first five in the initial comparison, and LightGBM added afterwards with the identical protocol. A seventh fit, the Optuna-tuned Random Forest, is covered in Section 4.

@@ metric_definitions
**Metric definitions.** **Observation-level Top-1 accuracy** (primary): for each GPS fix the highest-scoring candidate is compared with the aligned reference arc; the denominator is all fixes, so fixes without the true arc among their candidates count as errors. **Candidate-level macro F1**: every candidate row is a binary example (threshold 0.5) and the F1 of both classes is averaged. Candidate-level metrics are inflated by class imbalance (only about 16% of rows are positives) and depend on the threshold, so Top-1 is the primary selection metric: it is exactly the map-matching objective. **Route continuity**: share of consecutive fixes (same segment) whose predicted arcs are equal, directly connected or connected through one intermediate arc. For reference, choosing the nearest arc gives Top-1 {base_nearest} and the nearest arc whose direction agrees with the movement gives {base_heading}. Note: train Top-1 is computed only on fixes whose true arc is in the candidate set, so the train-minus-CV gaps below are inflated by up to about 0.6 points.

@@ LogisticRegression.A
**A. Model Understanding.** 1. Logistic regression is the natural linear baseline for a binary candidate classifier: it outputs calibrated-looking probabilities that can be used directly as HMM emissions, and its coefficients are interpretable (distance and heading agreement dominate). 2. Strengths: fast (training {LogisticRegression_train_time_s} s, inference {LogisticRegression_infer_ms_per_1000_obs} ms per 1000 fixes) and stable across folds. Limitation: the decision boundary is linear in the scaled features, so interactions such as "heading matters only when moving" cannot be represented.

@@ LogisticRegression.B
**B. Training and Configuration.** 5. StandardScaler, then L2-regularised logistic regression with C = 1.0, lbfgs solver, max_iter = 1000, random_state = 42, sample weights from label confidence. 6. Weighted binary cross-entropy (log loss) with an L2 penalty.

@@ LogisticRegression.C
8. Student-test observation-level Top-1 {LogisticRegression_test_top1}, candidate-level macro F1 {LogisticRegression_test_cand_macro_f1}, log loss {LogisticRegression_test_log_loss}. 9. No overfitting: train Top-1 {LogisticRegression_train_top1} vs CV {LogisticRegression_cv_top1_mean} (gap {LogisticRegression_gap} points, within the 0.6-point measurement caveat) and test {LogisticRegression_test_top1}. It shows mild underfitting instead: it is about one point below the tree ensembles in both CV and test, consistent with a linear model missing feature interactions.

@@ GaussianNB.A
**A. Model Understanding.** 1. Gaussian Naive Bayes is the simplest probabilistic (generative) model and shows how far class-conditional feature distributions alone can go. 2. Strengths: trains in {GaussianNB_train_time_s} s, no tuning needed. Limitation: it assumes features are independent given the class, which is clearly false here (dist, rel_dist and rank are strongly related), so its probabilities are over-confident.

@@ GaussianNB.B
**B. Training and Configuration.** 5. StandardScaler, then GaussianNB with var_smoothing = 1e-9 (default), sample weights from label confidence. 6. Maximum-likelihood estimates of per-class Gaussian means and variances; there is no iterative loss.

@@ GaussianNB.C
8. Student-test Top-1 {GaussianNB_test_top1}, macro F1 {GaussianNB_test_cand_macro_f1}, log loss {GaussianNB_test_log_loss}. 9. No overfitting (train Top-1 {GaussianNB_train_top1} vs CV {GaussianNB_cv_top1_mean}). Its ranking of candidates is about as good as logistic regression, but its log loss is the worst of all models ({GaussianNB_test_log_loss}), which is the expected effect of the violated independence assumption: probabilities are pushed towards 0 and 1. That makes it a poor emission model for the HMM.

@@ RandomForest.A
**A. Model Understanding.** 1. A Random Forest averages many decorrelated decision trees, so it captures non-linear interactions (for example distance thresholds that depend on heading agreement and speed) while staying robust to noisy labels. 2. Strengths: best student-test Top-1 among the six models and good probability quality (log loss {RandomForest_test_log_loss}). Limitations: the largest artefact (deep trees) and slower inference ({RandomForest_infer_ms_per_1000_obs} ms per 1000 fixes) than the linear and boosting models.

@@ RandomForest.B
**B. Training and Configuration.** 5. n_estimators = 150, max_depth = 20, min_samples_leaf = 5, max_features = sqrt, bootstrap, random_state = 42, sample weights from label confidence. 6. Each tree greedily minimises weighted Gini impurity; the forest averages the trees' class probabilities.

@@ RandomForest.C
8. Student-test Top-1 {RandomForest_test_top1}, macro F1 {RandomForest_test_cand_macro_f1}, log loss {RandomForest_test_log_loss}. 9. Moderate overfitting: train Top-1 {RandomForest_train_top1} vs CV {RandomForest_cv_top1_mean} (gap {RandomForest_gap} points), the largest gap of all models, as expected for depth-20 trees. It still generalises well: the student-test Top-1 ({RandomForest_test_top1}) is in line with CV, and CV is stable across folds (std {RandomForest_cv_top1_std}).

@@ HistGradientBoosting.A
**A. Model Understanding.** 1. Histogram-based gradient boosting builds shallow trees sequentially, each correcting the previous errors, which suits tabular data with interacting features. 2. Strengths: near the top on every metric, fast to train ({HistGradientBoosting_train_time_s} s) and compact. Limitation: more hyperparameters to tune, and without early stopping the number of iterations is fixed by hand.

@@ HistGradientBoosting.B
**B. Training and Configuration.** 5. max_iter = 200, learning_rate = 0.1, max_leaf_nodes = 31, early_stopping disabled (so no random row-level validation split leaks fixes of the same trip), random_state = 42, sample weights. 6. Binary log loss (cross-entropy), minimised by gradient boosting.

@@ HistGradientBoosting.C
8. Student-test Top-1 {HistGradientBoosting_test_top1}, macro F1 {HistGradientBoosting_test_cand_macro_f1}, log loss {HistGradientBoosting_test_log_loss}. 9. Slight overfitting only: train Top-1 {HistGradientBoosting_train_top1} vs CV {HistGradientBoosting_cv_top1_mean} (gap {HistGradientBoosting_gap} points) and test close to CV. Per-iteration loss was not recorded in these runs, so no training curve is shown.

@@ MLP.A
**A. Model Understanding.** 1. A shallow multilayer perceptron learns a smooth non-linear function of the features and gives probabilities through a sigmoid output, which fits the HMM emission role. 2. Strengths: fastest inference among the non-linear models ({MLP_infer_ms_per_1000_obs} ms per 1000 fixes) and the best route continuity before decoding. Limitations: slowest training ({MLP_train_time_s} s), sensitive to feature scaling and to the number of epochs.

@@ MLP.B
**B. Training and Configuration.** 5. StandardScaler, then hidden layers (64, 32), ReLU, Adam with learning rate 1e-3, L2 alpha = 1e-4, batch size 512, 40 epochs without early stopping, random_state = 42, sample weights. 6. Weighted binary cross-entropy plus L2 penalty, minimised by Adam.

@@ MLP.C
8. Student-test Top-1 {MLP_test_top1}, macro F1 {MLP_test_cand_macro_f1}, log loss {MLP_test_log_loss}. 9. No overfitting: train Top-1 {MLP_train_top1} vs CV {MLP_cv_top1_mean} (gap {MLP_gap} points), and in the per-epoch history (Fig 15) the validation log loss on held-out trips keeps falling from {mlp_val_first} to {mlp_val_last} together with the training loss ({mlp_tr_first} to {mlp_tr_last}). sklearn reported that 40 epochs had not fully converged, so a few more epochs could still help slightly.

@@ LightGBM.A
**A. Model Understanding.** 1. LightGBM is a gradient-boosting library that grows trees leaf-wise on histogram bins; it was added after the first comparison as a second boosting implementation. 2. Strengths: best CV macro F1 and CV Top-1, best student-test log loss ({LightGBM_test_log_loss}) and very fast training ({LightGBM_train_time_s} s). Limitation: leaf-wise growth can overfit small data if num_leaves is large, and it adds an extra dependency to the deployment.

@@ LightGBM.B
**B. Training and Configuration.** 5. n_estimators = 300, learning_rate = 0.05, num_leaves = 31, min_child_samples = 20, no row or column subsampling, random_state = 42, sample weights. 6. Binary log loss (cross-entropy), minimised by gradient boosting.

@@ LightGBM.C
8. Student-test Top-1 {LightGBM_test_top1}, macro F1 {LightGBM_test_cand_macro_f1}, log loss {LightGBM_test_log_loss}. 9. Slight overfitting only: train Top-1 {LightGBM_train_top1} vs CV {LightGBM_cv_top1_mean} (gap {LightGBM_gap} points), test close to CV. Per-iteration loss was not recorded in these runs.

@@ selection
All six models fall within about 1.2 points of Top-1 of each other, which shows the features carry most of the information. Ensembles (Random Forest, HistGradientBoosting, LightGBM) and the MLP clearly beat the linear and Naive Bayes models, and Naive Bayes has by far the worst probability quality. In the initial comparison the Random Forest had the highest student-test Top-1 ({RandomForest_test_top1}) and was carried forward. Because the classifier feeds the HMM, the final choice was made at system level on training data only (Section 4 and 5): untuned Random Forest {sel_rf_train}, LightGBM {sel_lgbm_train} and tuned Random Forest {sel_tun_train} train out-of-fold Top-1 after decoding. The untuned Random Forest is deployed. The margin to LightGBM is within noise, so either model would be a reasonable choice.

@@ hpo
The Random Forest was tuned with Optuna's TPE sampler (`experiments/05_optuna_tune.py`).

- **1. Hyperparameters searched:** n_estimators, max_depth, min_samples_leaf, max_features, class_weight.
- **2. Ranges:** n_estimators 50-300 (step 25), max_depth 6-30, min_samples_leaf 1-50 (log scale), max_features in (sqrt, 0.5, 1.0), class_weight in (None, balanced).
- **3. Objective:** mean observation-level Top-1 over GroupKFold(3) folds grouped by trip on the 77 train trips (fitting on at most 400 fixes per trip to keep the search affordable; scoring on all fixes of the held-out trips). Trial 0 was the untuned configuration ({opt_trial0_value}).
- **4. Trials:** {opt_trials} trials in {opt_minutes} minutes; the full trial table is in results/phase5/optuna_trials.csv.
- **5. Best values:** trial {opt_best_trial}: {opt_params} (CV(3) Top-1 {opt_best_value}).
- **6. Did tuning help?** Only marginally for the classifier and not for the system (Table 5).

@@ hpo_after
The tuned model gained {tun_gain_cv} points of CV Top-1 and {tun_gain_test} points of student-test Top-1, but lost on macro F1 ({tun_test_cand_macro_f1} vs {unt_test_cand_macro_f1}), log loss ({tun_test_log_loss} vs {unt_test_log_loss}) and speed. The search chose class_weight = balanced, which helps ranking slightly but distorts the probabilities. The HMM uses these probabilities as emissions, so after decoding the tuned model is worse on train ({sel_tun_train} vs {sel_rf_train}) and on student-test ({sel_tun_test} vs {sel_rf_test}). The untuned Random Forest was therefore kept. Lesson: the optimisation objective should be the system-level metric (Top-1 after decoding), not the classifier-only Top-1.

@@ hmm
`src/hmm.py` implements the decoder; decoder parameters were tuned on train out-of-fold probabilities (`experiments/04_hmm_tune_compare.py`) and checked on student_test.

- **1. Hidden states:** the candidate arcs of each fix. Each state keeps its arc id and the position where the fix projects onto the arc, so every state corresponds to exactly one road segment.
- **2. Emissions:** the classifier probability, normalised over the fix's candidates, raised to the power alpha (log emission = alpha x log p).
- **3. Transitions:** log P(b given a) = -|route distance - straight distance| / beta. The route distance is the distance driven along the directed network from the projection on arc a to the projection on arc b (rest of a + shortest path + part of b), found with a bounded Dijkstra search (cutoff = min(50 m/s x time gap + 150 m, 3000 m)) and cached per node. Staying on the same arc, a direct continuation, multi-arc paths and U-turns onto the reverse twin (penalised) are all handled.
- **4. Missing candidates and gaps:** a fix with no candidate within 30 m (and none at 100 m) is marked unmatched and skipped; the chain links the fixes around it. A time gap above 60 s starts a new segment and the decoder restarts.
- **5. Disconnected segments:** transitions beyond the cutoff are impossible (probability 0). If no transition into a fix is possible at all, the step is retried with a 10 km cutoff, and if that also fails the decoder restarts from the best previous state with a fixed penalty. Both cases are tagged relaxed and counted; the decoder never crashes.
- **6. Did the HMM improve the route?** Yes, clearly (Table 6).

Decoder tuning: grid over alpha ({dec_grid_alpha}), beta ({dec_grid_beta}) and the U-turn penalty ({dec_grid_rp}), {dec_grid_n} combinations, objective = train out-of-fold Top-1 after decoding. Best: alpha = {dec_alpha}, beta = {dec_beta}, penalty = {dec_rp} with {dec_train_hmm} (classifier alone {dec_train_clf}). Because Viterbi's choice does not change when all scores are divided by alpha, only alpha x beta and penalty / alpha matter, which is why several grid points score almost the same.

@@ hmm_after
The decoder improved Top-1 on {n_improved} of {n_trips_st} trips; the largest gain was {best_gain_trip}. Route continuity rose to {hmm_route_continuity_trip_macro} and the edit distance to the reference route fell from {clf_norm_edit_distance_trip_macro} to {hmm_norm_edit_distance_trip_macro}. On student_test the decoder used {t_direct} direct, {t_multi} multi-arc, {t_rev} U-turn and {t_relaxed} relaxed transitions, with {t_unmatched} unmatched fixes and {t_starts} segment starts. The cost is runtime: about {hmm_runtime_ms_trip_macro} ms per trip instead of {clf_runtime_ms_trip_macro} ms.

**Where the HMM did not help:** Top-1 got worse on {worse_trips}. Most remaining errors (Table 8) are adjacent arcs at junctions ({err_adjacent_arc_pct} of errors: the predicted arc touches the reference arc, often just a boundary decision near a node) and direction flips ({err_direction_flip_pct}; {flip_77} in trip 00000077, the student-test trip with the largest share of medium-confidence labels in Phase 1, and {flip_07} in 00000007). Candidate misses ({err_candidate_miss_pct}) cannot be fixed by the decoder at all. The labels are produced by a distance-based alignment, and the transition model also uses distances, so part of the agreement is built in; accuracy should be read against reference labels that are not perfect ground truth.

@@ folium
Maps are generated by code (`experiments/00_eda.py` for EDA, `experiments/06_final_analysis.py` for the results) and committed as interactive HTML in results/. Each map shows the original GPS observations, the road network within 150 m of the track, the reference route (offline analysis only) and the predicted route, with distinct colours, a legend, layer control and a zoom fitted to the trip. The final maps colour each wrong fix by error category. The three worst student-test trips were mapped: {worst_trips}.

@@ folium_after
In trip 00000009 the predicted route follows the reference route along the main road. Adjacent-arc errors are spread along the route near junctions, while candidate-miss and parallel-road errors cluster at the start of the trip inside a village (Fig 24), where the predicted route takes a nearby street that differs from the reference route.

@@ deploy
**Laptop 1 (model and API)** runs `run_api.bat` (FastAPI + uvicorn on 0.0.0.0:8000, CORS enabled). At start-up it loads the model, the feature contract and the decoder configuration from `artifacts/`; `DATA_DIR` points at the folder of trips (the evaluator folder at final evaluation). Endpoints: `/health`, `/trips`, `/predict/(trip_id)`. A prediction request runs the full pipeline (preprocess, candidates, features, classifier, HMM) without reading `.route`; afterwards a separate evaluation module reads `.route`, aligns it and returns numbers only (Top-1, route continuity, candidate miss rate). If a trip has no `.route`, these metrics are null. Every request is logged on the console with the client address, trip id, status and latency.

**Laptop 2 (UI only)** serves a static HTML/JavaScript page (`run_ui.bat`, `python -m http.server 5500`) with Leaflet stored locally, so it works without a CDN. The user enters Laptop 1's address, the page lists the trips from `/trips`, and clicking a trip calls `/predict` and draws the GPS points, the predicted route, low-confidence and fallback points, a legend, the metrics and the events. Laptop 2 holds no model, data or ground truth.

@@ deploy_after
Tests: all {api_ok} of {api_n} student-test trips returned status ok through the API (client latency {api_lat_min} to {api_lat_max} ms per trip), with the same Top-1 values as the offline evaluation. As a robustness test the API was also run on all {rob_n} train trips: {rob_ok} returned ok; the slowest was {rob_slow_trip} ({rob_slow_s} s, {rob_slow_relaxed} relaxed transitions and {rob_slow_unmatched} unmatched fixes, the trip that is mostly off the network). A trip without `.route` returned null metrics without errors.

@@ integrity
- No trip appears in two partitions; cross-validation is grouped by trip; preprocessing and models are fitted on training rows only; the evaluator partition was never used. These are checked by assertions in `experiments/07_integrity_checks.py` (all passed).
- The prediction code never reads `.route`; the API response contains no reference labels or reference route; nothing is hardcoded.
- All numbers and figures in this report are produced by the code in the repository from the actual data (this PDF is built by `report/build_report.py` from results/).
- AI assistance: <describe here, consistent with AI_DISCLOSURE.md>.
