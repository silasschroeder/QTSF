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
  - (look at litlogger for proper metric & artifact handling); CHANGE: look at any metric visualization tool (TensorBoard) ✅
LOGS: `tensorboard --logdir rossmann-store-sales/rossmann-solution-silas/lightning_logs`

- 26.02.2026:
  - create QPatchTST ✅
  - create QTimesNet ✅
  - create QNHITS ✅


- Find reason for why exactly those 4 models
  - Understand those 4 models
  - Understand the changes made within the model structure
  - Illustrate changes (or make it possible to explain based on an illustration of the base models)
- Ask Jaakob/Pallavi for advice on the structure of quantum circuit


cross validation
hpo
early stopping

