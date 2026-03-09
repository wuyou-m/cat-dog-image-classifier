# test_model.py

from src.model import build_baseline_cnn, build_transfer_model, compile_model

baseline = build_baseline_cnn()
baseline = compile_model(baseline)
baseline.summary()

transfer = build_transfer_model()
transfer = compile_model(transfer)
transfer.summary()