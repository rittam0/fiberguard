"""Inference contract with a model stub; no raw benchmark or registry required."""
import importlib
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
from httpx import ASGITransport, AsyncClient


class InferenceContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_raw_probability_routing_and_portable_registry_path(self):
        model = Mock()
        model.predict_proba.return_value = np.array([[0.99, 0.01, 0, 0]])
        with patch("mlflow.set_tracking_uri") as tracking, patch("mlflow.xgboost.load_model", return_value=model) as load:
            api = importlib.import_module("fiberguard.api")
            app = api.create_app()
            manifest = json.loads(api.MANIFEST_PATH.read_text())
            tracking.assert_called_with(f"sqlite:///{(api.ROOT / manifest['mlflow_tracking_path']).resolve().as_posix()}")
            load.assert_called_with(manifest["registered_model_uri"])
        client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
        self.addAsyncCleanup(client.aclose)
        data = dict(lp_length_km=514, laser_current_ma=40, lp_power_dbm=-2, osnr_db=24, ber_db=-267)
        result = await client.post("/predict", json=data)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["decision"], "confident")
        self.assertEqual(result.json()["predicted_state"], "healthy/no-failure")
        np.testing.assert_equal(model.predict_proba.call_args.args[0], [[514, 40, -2, 24, -267]])
        model.predict_proba.return_value = np.array([[0.98, 0.02, 0, 0]])
        self.assertEqual((await client.post("/predict", json=data)).json()["decision"], "review")
        self.assertEqual((await client.post("/predict", json={**data, "unexpected": 1})).status_code, 422)
        self.assertEqual((await client.post("/predict", json={})).status_code, 422)
        self.assertEqual((await client.get("/health")).json()["model_version"], manifest["model_version"])
