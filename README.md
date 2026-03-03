Competition Goal:
Forecast 28 days into the future
"Note, that we have been warned that most of the time series contain zero values."
The data comprises 3049 individual products from 3 categories and 7 departments, sold in 10 stores in 3 states.

```
                         Total
                           │
                ┌──────────┼──────────┐
                │          │          │
               CA         TX         WI
                │          │          │
         ┌──────┼──────┐   ...       ...
         │      │      │
       CA_1   CA_2   CA_3
         │
   ┌─────┼────────────────────┐
   │     │                    │
 FOODS HOBBIES           HOUSEHOLD
   │
 FOODS_1
   │
 ITEM_ID
   │
 ITEM_ID × STORE_ID
```

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

TODO:
- Sync Jaakob: Are Quantum Devices available? If not so, contact Sarah
- Laufzeiten: Trainingszeit pro Epoche, Gesamttrainingszeit, Inferenzzeit
- Parameter Fairness: Parameteranzahl, `sum(p.numel() for p in model.parameters())`
- Naive Baseline: Last Value Forecast, Seasonal Naive Forecast
- Diebold-Mariano Test oder gepaarter t-Test auf Fehler (Sind unterschiede signifikant oder nur Zufall)

- Maybe: Adjust Circuit
- Maybe: Adjust circuit incoorperation within models
- Find reason for why exactly those 4 models
  - Understand those 4 models
  - Understand the changes made within the model structure
  - Illustrate changes (or make it possible to explain based on an illustration of the base models)
- Ask Jaakob/Pallavi for advice on the structure of quantum circuit

Big TODO for thesis:
- measuring one qubit exactly once may not lead to a value that is the most common/optimal one. 
- how often is the measuring done?
- how often should the measuring be done?