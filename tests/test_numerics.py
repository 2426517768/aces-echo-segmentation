"""Precision policy/resume tests use generated software fixtures only."""
from __future__ import annotations
from contextlib import nullcontext
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from aces import engine
from aces.data import RecordStore
from aces.models import METHODS, build_model
from aces.numerics import (NumericalError, effective_amp, precision_record, require_finite,
                           require_finite_state, require_finite_gradients)
from aces.synthetic import make_synthetic
from aces.utils import atomic_torch_save, read_json, write_json, load_checkpoint, digest, sha256


def cfg():
    value = read_json(Path(__file__).parents[1]/"settings.json")["training"]
    value.update(image_size=64, base_channels=8, graph_grid=4, graph_dim=32, graph_heads=4,
                 epochs=2, steps_per_epoch=2, warmup_epochs=0, ramp_epochs=1,
                 micro_batch=1, effective_batch=2, checkpoint_every_updates=1, cpu_threads=2, amp=True)
    return value


class NumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_segformer_fp32_policy_does_not_modify_requested_configuration(self):
        settings = cfg(); original = copy.deepcopy(settings)
        self.assertFalse(effective_amp("segformer_b0", settings, torch.device("cuda")))
        for method in METHODS:
            if method != "segformer_b0":
                self.assertTrue(effective_amp(method, settings, torch.device("cuda")))
            self.assertFalse(effective_amp(method, settings, torch.device("cpu")))
        self.assertEqual(settings, original)

    def test_real_fp16_overflow_fixture_is_finite_in_fp32(self):
        x = torch.full((2,2), 1000.)
        with torch.autocast("cpu", dtype=torch.float16):
            half = x@x
        full = x@x
        self.assertFalse(torch.isfinite(half).all())
        require_finite(full, "fixture")
        with self.assertRaises(NumericalError):require_finite(half, "fixture")

    def test_disabled_cuda_autocast_is_explicit_not_an_inherited_context(self):
        with patch.object(torch, "autocast", return_value=nullcontext()) as context:
            with engine.autocast(torch.device("cuda"), False):pass
        self.assertFalse(context.call_args.kwargs["enabled"])

    def test_precision_transition_records_legacy_epoch_and_update(self):
        settings=cfg();settings["epochs"]=200
        legacy={"epoch":87,"next_step":50,"scaler":{"scale":65536.}}
        record=precision_record("segformer_b0", settings, torch.device("cuda"), legacy)
        self.assertEqual(record["active_precision"],"float32")
        self.assertEqual(record["precision_history"][0]["precision"],"float16_amp")
        self.assertEqual((record["precision_history"][1]["epoch"],record["precision_history"][1]["update"]),(88,51))
        resumed={"epoch":87,"next_step":75,"scaler":{},"precision_history":record["precision_history"]}
        again=precision_record("segformer_b0",settings,torch.device("cuda"),resumed)
        self.assertEqual(again["precision_history"],record["precision_history"])

    def test_gradient_check_does_not_clip_or_replace_gradients(self):
        model=torch.nn.Linear(3,2)
        for p in model.parameters():p.grad=torch.full_like(p,10000.)
        original=[p.grad.clone() for p in model.parameters()]
        require_finite_gradients(model)
        self.assertTrue(all(torch.equal(a,p.grad) for a,p in zip(original,model.parameters())))
        model.weight.grad[0,0]=float("nan")
        with self.assertRaises(NumericalError):require_finite_gradients(model)
        self.assertTrue(torch.isnan(model.weight.grad[0,0]))

    def test_state_guard_identifies_corrupt_optimizer_tensor(self):
        state={"optimizer":{"state":[{"exp_avg":torch.tensor([float("inf")])}]}}
        with self.assertRaises(NumericalError) as error:require_finite_state(state,"checkpoint")
        self.assertEqual(error.exception.details["state_path"],"/optimizer/state/0/exp_avg")

    def legacy_fixture(self,root,method="segformer_b0"):
        manifest=make_synthetic(root/"data")
        settings=cfg();run=root/"run"
        save=engine.atomic_torch_save
        def stop_after_epoch(state,path):
            save(state,path)
            if Path(path).name=="last.pt" and state["epoch"]==1 and state["next_step"]==0:
                raise KeyboardInterrupt()
        with patch.object(engine,"atomic_torch_save",side_effect=stop_after_epoch):
            with self.assertRaises(KeyboardInterrupt):engine.train(manifest,run,method,settings,17,.25,False)
        for name in ["last.pt","best.pt"]:
            state=load_checkpoint(run/name)
            state.pop("numerical_precision",None);state.pop("precision_history",None)
            if name=="last.pt":
                state["scaler"]={"scale":65536.,"growth_factor":2.,"backoff_factor":.5,
                                 "growth_interval":2000,"_growth_tracker":17}
            atomic_torch_save(state,run/name)
        (run/"numerical_precision.json").unlink(missing_ok=True)
        return manifest,settings,run

    def test_legacy_resume_preserves_contract_history_and_finishes(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest,settings,run=self.legacy_fixture(Path(folder))
            old=load_checkpoint(run/"last.pt");contract=read_json(run/"contract.json")
            engine.train(manifest,run,"segformer_b0",settings,17,.25,False)
            new=load_checkpoint(run/"last.pt")
            self.assertEqual(new["epoch"],2)
            self.assertEqual(new["contract_hash"],old["contract_hash"])
            self.assertEqual(read_json(run/"contract.json"),contract)
            self.assertEqual(new["history"][:1],old["history"])
            self.assertEqual(new["scaler"],{})
            self.assertEqual(new["precision_history"][-1]["precision"],"float32")
            self.assertTrue(read_json(run/"training_complete.json")["status"]=="completed")

    def test_revalidation_preserves_best_weights_and_training_rng(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest,settings,run=self.legacy_fixture(Path(folder))
            previous_best=load_checkpoint(run/"best.pt")
            previous_last=load_checkpoint(run/"last.pt")
            last_hash=sha256(run/"last.pt")
            save=engine.atomic_torch_save
            def stop_after_revalidation(state,path):
                save(state,path)
                if Path(path).name=="best.pt":raise KeyboardInterrupt()
            with patch.object(engine,"atomic_torch_save",side_effect=stop_after_revalidation):
                with self.assertRaises(KeyboardInterrupt):engine.train(manifest,run,"segformer_b0",settings,17,.25,False)
            current=load_checkpoint(run/"best.pt")
            self.assertTrue(all(torch.equal(v,current["model"][k]) for k,v in previous_best["model"].items()))
            self.assertEqual(sha256(run/"last.pt"),last_hash)
            self.assertTrue(torch.equal(torch.get_rng_state(),previous_last["torch_rng"]))

    def test_nonfinite_loss_preserves_safe_checkpoint_and_writes_case_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest,settings,run=self.legacy_fixture(Path(folder))
            previous=sha256(run/"last.pt")
            with patch.object(engine,"supervised_loss",return_value=torch.tensor(float("nan"))):
                with self.assertRaises(NumericalError):engine.train(manifest,run,"segformer_b0",settings,17,.25,False)
            self.assertEqual(previous,sha256(run/"last.pt"))
            diagnostic=read_json(run/"numerical_error.json")
            self.assertEqual(diagnostic["stage"],"training loss")
            self.assertTrue(diagnostic["case_ids"])
            self.assertFalse((run/"training_complete.json").exists())

    def test_corrupt_checkpoint_is_rejected_before_any_optimizer_update(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest,settings,run=self.legacy_fixture(Path(folder))
            state=load_checkpoint(run/"last.pt")
            tensor=next(x for x in state["student"].values() if x.is_floating_point())
            tensor.reshape(-1)[0]=float("inf")
            atomic_torch_save(state,run/"last.pt");previous=sha256(run/"last.pt")
            with patch.object(torch.optim.AdamW,"step",side_effect=AssertionError("Invalid checkpoint was used")):
                with self.assertRaises(NumericalError):engine.train(manifest,run,"segformer_b0",settings,17,.25,False)
            self.assertEqual(previous,sha256(run/"last.pt"))
            self.assertEqual(read_json(run/"numerical_error.json")["stage"],"checkpoint")

    def test_corrupt_gradient_is_rejected_without_overwriting_safe_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest,settings,run=self.legacy_fixture(Path(folder))
            previous=sha256(run/"last.pt")
            original=engine.build_model
            def model_with_bad_grad(*args,**kw):
                model=original(*args,**kw)
                next(model.parameters()).register_hook(lambda grad:torch.full_like(grad,float("inf")))
                return model
            with patch.object(engine,"build_model",side_effect=model_with_bad_grad):
                with self.assertRaises(NumericalError):engine.train(manifest,run,"segformer_b0",settings,17,.25,False)
            self.assertEqual(previous,sha256(run/"last.pt"))
            self.assertEqual(read_json(run/"numerical_error.json")["stage"],"gradients")

    def test_bad_optimizer_state_is_not_saved_as_a_safe_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest,settings,run=self.legacy_fixture(Path(folder))
            previous=sha256(run/"last.pt")
            original=torch.optim.AdamW.step
            def poisoned_step(optimizer,*args,**kwargs):
                result=original(optimizer,*args,**kwargs)
                next(iter(optimizer.state.values()))["exp_avg"].reshape(-1)[0]=float("inf")
                return result
            with patch.object(torch.optim.AdamW,"step",poisoned_step):
                with self.assertRaises(NumericalError):engine.train(manifest,run,"segformer_b0",settings,17,.25,False)
            self.assertEqual(previous,sha256(run/"last.pt"))
            self.assertEqual(read_json(run/"numerical_error.json")["stage"],"checkpoint before save")

    def test_nan_predictions_are_not_converted_into_validation_or_test_scores(self):
        class BadModel(torch.nn.Module):
            def forward(self,x,view):return {"logits":torch.full((len(x),4,*x.shape[-2:]),float("nan"))}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=make_synthetic(root/"data");settings=cfg()
            store=RecordStore(manifest,64)
            with self.assertRaises(NumericalError):engine.validation_score(BadModel(),store,torch.device("cpu"))
            with self.assertRaises(NumericalError):engine.collect_temperature_samples(BadModel(),store,torch.device("cpu"),64)
            state={"contract":{"config":settings,"method":"segformer_b0","manifest_hash":digest(read_json(manifest))}}
            with patch.object(engine,"load_model_for_evaluation",return_value=(BadModel(),state)):
                with self.assertRaises(NumericalError):engine.evaluate(root/"unused.pt",manifest,root/"test",require_cuda=False)
            self.assertFalse((root/"test/summary.json").exists())


if __name__=="__main__":unittest.main(verbosity=2)
