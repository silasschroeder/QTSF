Goals:
- 25.02.2026:
  - Include more models out of neural forecast ✅
  - Include py files of neural forecast models ✅
  - Learn where to modify py files for qml ✅
  - Learn how to create modification ✅
  - (look at litlogger for proper metric & artifact handling); Currently W&B, Previously TensorBoard ✅

- 26.02.2026:
  - create QPatchTST ✅
  - create QTimesNet ✅
  - create QNHITS ✅

- 02.03.2026
  - hpo (implemented via Optuna) ✅
  - Random Seeds & Reproducibility ✅
  - Evaluation Metrics: MAE RMSE MAPE sMAPE (Optional:MASE) ✅
  - cross validation (Rolling forecast origin evaluation / time series cross validation) ✅
  - early stopping ✅
  - device: ✅
    - check how model currently is trained (noise free, nisq simulation, nisq) ✅
    - create additional for loop with training devices ✅

05.03.2026:
- Maybe use MLFlow instead of W&B ✅
- Laufzeiten: Trainingszeit pro Epoche, Gesamttrainingszeit, Inferenzzeit ✅
- Parameter Fairness: Parameteranzahl, `sum(p.numel() for p in model.parameters())` ✅
- Naive Baseline: Last Value Forecast ✅
- learn how to use aer (https://pennylane.ai/qml/demos/tutorial_how_to_import_qiskit_noise_models & https://pennylane.ai/qml/demos/tutorial_noisy_circuits) ❌ (pennylane integration of qiskit unstable)

10.03.2026:
- Fulfill business understanding ✅
- Adjust Circuit based on research papers ✅

11.03.2026:
- Solve problem: Predictions can only be made given the customers ✅
- Save models ✅
- Make models include newer data (nf.predict(df=…)) ✅

12.03.2026:
- rethink model selection based on https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/overview.html

Later:
- Optimize Parameters of Quantum Circuit(s)
- Include visualizations into model presentation
- Understand DLinear ✅
- Create Visualisation DLinear
- Understand NHITS ✅
- Create Visualisation NHITS
- Understand PatchTST
- Create Visualisation PatchTST
- Understand TimesNet
- Create Visualisation TimesNet
- Find reason for why exactly those 4 models
  - Understand those 4 models
  - Understand the changes made within the model structure
  - Illustrate changes (or make it possible to explain based on an illustration of the base models)

Big TODO for thesis:
- measuring one qubit exactly once may not lead to a value that is the most common/optimal one. 
- how often is the measuring done?
- how often should the measuring be done?
- How is retraining handled? Can new data be simply included? Is there an option to parse it in the predict function for context?
- Shap analysis (which feature is important, which not)

New Goals:
- Understand how models are constructed? Are there better positions for circuit? 
- Circuit adjustment
- Noise simulation sadly didnt work with IBM tools
- Writing thesis

Additional Information:
Model	    hist_exog futr_exog	stat_exog
NHITS	    ✅	       ✅	      ✅
TimesNet	❌	       ✅	      ❌
DLinear	  ❌	       ❌	      ❌
PatchTST	❌	       ❌	      ❌