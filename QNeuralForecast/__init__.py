"""
QNeuralForecast - Quantum-augmented time series forecasting models.

This module registers custom quantum models with NeuralForecast's save/load system.
"""

from .dlinear import QDLinear
from .nhits import QNHITS
from .deepar import QDeepAR
from .timesnet import QTimesNet

# Register quantum models with NeuralForecast's MODEL_FILENAME_DICT
# This enables nf.save() and NeuralForecast.load() to work with quantum models
from neuralforecast.core import MODEL_FILENAME_DICT

MODEL_FILENAME_DICT["qdlinear"] = QDLinear
MODEL_FILENAME_DICT["qnhits"] = QNHITS
MODEL_FILENAME_DICT["qdeepar"] = QDeepAR
MODEL_FILENAME_DICT["qtimesnet"] = QTimesNet

__all__ = ["QDLinear", "QNHITS", "QDeepAR", "QTimesNet"]
