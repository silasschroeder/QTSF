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
- rethink model selection based on https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/overview.html ✅
- individualize the HPO for each model due to different architectures ✅
- Do Data Understanding (watch kaggle and recreate script) ✅

13.03.2026:
- Check if metrics get logged onto mlflow ✅
- Let results run on Databricks ✅

16.03.2026:
- Solve ValueError: Model QDLinear is not supported for saving. (probably for all q models) ✅
- include qulacs as a noise simulator (research done but didnt work) ✅
- analyze behaviour of input variables in qlstm and my models (write about it in modeling) ✅
- Theoretical Background ✅

18.03.2026
- Fix Pipeline (Simulator not working) --> does work but takes long ✅

23.03.2026
- Include IBM API --> (testwise done) ✅
- adjust circuit ✅
- lower parameters due to long compute times with simulator ✅

Later:
- write modeling
- transfer github
- Optimize Parameters of Quantum Circuit(s)
- Include visualizations into model presentation
- Understand DLinear ✅
- Create Visualisation DLinear
- Understand NHITS ✅
- Create Visualisation NHITS
- Understand DeepAR
- Create Visualisation DeepAR
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

New Goals:
- Understand how models are constructed? Are there better positions for circuit? 
- Noise simulation sadly didnt work with IBM tools
- Writing thesis

Additional Information:
Model	    hist_exog futr_exog	stat_exog
DLinear	  ❌	       ❌	      ❌
PatchTST	❌	       ❌	      ❌ <-- removed from project
TimesNet	❌	       ✅	      ❌
DeepAR    ❌        ✅        ✅
NHITS	    ✅	       ✅	      ✅