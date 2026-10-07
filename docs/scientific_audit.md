# Scientific Audit — Lorenz–96 Data Assimilation with EnKF

## Verdict summary

The project asks a correct and standard question: how finite-ensemble stochastic EnKF behaves on chaotic Lorenz–96 state estimation under varying ensemble size, observations, localization and inflation.

**No fundamental implementation or method error was identified in this pass.** The severe small-ensemble failures are scientifically plausible finite-ensemble covariance/under-dispersion failures, not evidence that the EnKF code is broken.

## 1. Dynamical model

Lorenz–96 is a deliberately idealised chaotic test system, not a physical atmosphere model. It is appropriate for studying filtering mechanisms, ensemble-size effects and localization in a controlled nonlinear setting.

Any conclusions should therefore be stated as EnKF/L96 conclusions, not universal numerical weather-prediction results.

## 2. Stochastic EnKF update

The implementation uses forecast anomalies with the (1/(N_e-1)) covariance normalization, computes state-observation and observation-space sample covariances, and perturbs each observation independently according to the stated observation covariance.

This is the standard stochastic/perturbed-observation EnKF formulation.

## 3. Independent validation

The repository compares the EnKF against an exact Kalman filter on a separate linear-Gaussian system. As ensemble size increases, both ensemble mean and covariance approach the exact Kalman quantities.

This is important because good Lorenz–96 RMSE alone would not independently verify the analysis update.

## 4. Localization

Gaspari–Cohn tapering uses cyclic distance appropriate to the Lorenz–96 ring. For the direct observation operator used here, the state-observation and observation-observation tapers are consistent with applying the same spatial localization idea to the sampled covariance blocks.

Localization is not enabled merely to improve headline numbers; unlocalized results and small-ensemble failures are retained.

## 5. Inflation

Multiplicative variance inflation rescales forecast anomalies by the square root of the requested covariance inflation factor. This is a standard convention and the implementation/test agree on that convention.

## 6. Failure regimes

The full small-ensemble experiment with localization still gives large time-averaged analysis RMSE for (N_e=4,6,8), while (N_e=10) improves substantially. The ensemble remains numerically finite rather than exploding.

This is a useful distinction:

- **numerical divergence:** NaN/unstable integration;
- **filter divergence/failure:** an apparently stable filter that remains persistently far from truth and under-dispersed.

The experiment demonstrates the second phenomenon.

## 7. Repeated-seed evidence

Twenty-seed results show the unlocalized (N_e=20) filter performing very poorly while localization greatly reduces mean RMSE, though with one notably worse seed. This supports reporting a distribution across seeds rather than a single favourable trajectory.

The rank histogram is U-shaped rather than flat, consistent with residual under-dispersion/miscalibration. The repository correctly calls it a diagnostic rather than proof of calibration.

## 8. Qualifications

- The truth and ensemble use the same Lorenz–96 model in the principal experiments; structural model error is therefore absent.
- Observation errors are independent and Gaussian.
- Localization radius and inflation are empirical tuning parameters.
- A stochastic EnKF adds Monte Carlo noise from perturbed observations; deterministic square-root filters could be a useful comparison but are not required to make the present method valid.
- Low RMSE does not by itself establish probabilistic calibration.

## Wrong-problem / wrong-method classification

- **Wrong problem?** No.
- **Wrong method?** No.
- **Implementation defect found?** None in the audited EnKF update/localization/observation logic.
- **Negative results to preserve?** Yes: small-ensemble failure, under-dispersion and seed sensitivity are meaningful findings.
- **Claims needing qualification?** Results are specific to perfect-model Lorenz–96 and the tested tuning/observation regimes.

## Final verdict

**Supported project with appropriate negative-result reporting.** No code correction is proposed in this audit PR.
