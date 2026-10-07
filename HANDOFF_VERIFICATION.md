# Handoff: Lorenz-96 EnKF verification

## Current scientific status
The Lorenz-96/EnKF study is scientifically sound for the tested idealized perfect-model setting. The stochastic EnKF covariance normalization, perturbed observations, cyclic Gaspari-Cohn localization and multiplicative inflation convention are appropriate. An independent linear-Gaussian Kalman test validates the update. Small-ensemble failure and U-shaped rank histograms are genuine filter-divergence/under-dispersion findings, not numerical-integrator failures.

## Local work
Run all tests, especially exact Kalman validation and localization tests. Reproduce baseline, repeated-seed, ensemble-size, localization, inflation, observation-density/frequency/noise and failure-regime studies. Report distributions across seeds, not only a favorable seed. Keep free-forecast/chaos evidence separate from assimilation performance. State explicitly that conclusions are for Lorenz-96 and primarily perfect-model experiments.

## Claude prompt
Continue the scientific audit of lorenz96-data-assimilation. Read this handoff, docs/scientific_audit.md, tests, configs and committed result files. Run the full suite and independent Kalman validation before rerunning studies. Reproduce multi-seed experiments and quantify RMSE/spread/rank-histogram behavior across ensemble size, localization and inflation. Preserve small-ensemble failure as a genuine negative result. Distinguish filter divergence from numerical ODE instability and calibration from accuracy. Do not generalize Lorenz-96 results to operational atmospheric DA. Show old/new results and changed files before commits; do not merge or force-push.