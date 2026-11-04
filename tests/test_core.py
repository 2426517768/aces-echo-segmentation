from __future__ import annotations
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from aces.data import (RecordStore, annotation_subset, validate_manifest, letterbox, native_logits, read_array,
                       prepare_camus, prepare_external, safe_extract)
from aces.engine import train, fit_temperature, evaluate, benchmark
from aces.losses import supervised_loss, unsupervised_loss, update_teacher, pseudo_targets
from aces.metrics import binary_metrics, aggregate_case_metrics, probability_metrics, aggregate_calibration, paired_bootstrap
from aces.models import build_model, METHODS
from aces.prior import AnatomicalPrior, mask_features
from aces.synthetic import make_synthetic
from aces.utils import read_json, write_json, load_checkpoint
from aces.report import export_report


def tiny_cfg():
    cfg = read_json(Path(__file__).parents[1]/"settings.json")["training"]
    cfg.update(image_size=64, base_channels=8, graph_grid=4, graph_dim=32, graph_heads=4,
               epochs=2, steps_per_epoch=2, warmup_epochs=0, ramp_epochs=1,
               micro_batch=1, effective_batch=2, log_every_updates=10, checkpoint_every_updates=1, amp=False)
    return cfg


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_graph_forward_backward_and_parameter_paths(self):
        cfg = tiny_cfg()
        for method in [m for m in METHODS if m != "segformer_b0"]:
            model = build_model(method, cfg)
            x = torch.rand(2, 1, 64, 64)
            target = torch.randint(4, (2, 64, 64))
            out = model(x, torch.tensor([0, 1]))
            self.assertEqual(out["logits"].shape, (2, 4, 64, 64))
            loss = supervised_loss(out, target)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            if model.graph:
                self.assertIsNotNone(model.graph.bias.grad)
                self.assertGreater(model.graph.embed.weight.grad.abs().sum().item(), 0)

    def test_segformer_native_implementation_shape(self):
        try:
            import transformers
        except ImportError:
            self.skipTest("Optional transformers not installed")
        model = build_model("segformer_b0", tiny_cfg()).eval()
        with torch.no_grad():
            out = model(torch.rand(1, 1, 64, 64), torch.tensor([1]))
        self.assertEqual(tuple(out["logits"].shape), (1, 4, 64, 64))

    def test_empty_pseudo_label_loss_is_zero_and_finite(self):
        logits = torch.randn(1, 4, 16, 16, requires_grad=True)
        loss = unsupervised_loss(logits, torch.zeros(1,16,16,dtype=torch.long), torch.zeros(1,16,16))
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertEqual(logits.grad.abs().sum().item(), 0)

    def test_nonzero_unsupervised_gradients(self):
        logits = torch.zeros(1, 4, 8, 8, requires_grad=True)
        loss = unsupervised_loss(logits, torch.ones(1,8,8,dtype=torch.long), torch.ones(1,8,8))
        loss.backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_teacher_ema_is_detached(self):
        model = torch.nn.Linear(3, 2)
        teacher = copy.deepcopy(model)
        old = teacher.weight.detach().clone()
        with torch.no_grad():
            model.weight.add_(1)
        update_teacher(teacher, model, 0.9)
        self.assertTrue(torch.allclose(teacher.weight, old+0.1))
        self.assertIsNone(teacher.weight.grad)

    def test_prior_bounded_missing_class_and_positive_pseudo_path(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = make_synthetic(d)
            store = RecordStore(manifest, 64)
            _, labeled = annotation_subset(store.manifest["records"], 0.5)
            prior = AnatomicalPrior.fit(labeled, store.canonical)
            _, mask, meta = store.canonical(labeled[0])
            v = torch.tensor([meta["view_id"]])
            s = prior.score(mask[None], v)
            self.assertTrue(0 < s.item() <= 1)
            self.assertEqual(prior.score(torch.zeros_like(mask)[None], v).item(), 0)
            bad = copy.deepcopy(labeled)
            bad[0]["split"] = "test"
            with self.assertRaises(ValueError):
                AnatomicalPrior.fit(bad, store.canonical)
            class Oracle(torch.nn.Module):
                def forward(self, x, view):
                    logits = torch.nn.functional.one_hot(mask, 4).permute(2,0,1).float()[None]*20
                    return {"logits": logits.expand(x.shape[0],-1,-1,-1)}
            image, _, _ = store.canonical(labeled[0])
            strong, pseudo, weights, score = pseudo_targets(Oracle(), image[None], v, prior, "confidence", confidence=0.9)
            self.assertGreater((weights > 0).sum().item(), 0)
            self.assertFalse(weights.requires_grad)
            self.assertFalse(strong.requires_grad)

    def test_patient_split_leakage_and_hidden_labels(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = make_synthetic(d)
            store = RecordStore(manifest)
            record = store.manifest["records"][0]
            self.assertIsNone(store.canonical(record, with_mask=False)[1])
            self.assertIsNotNone(store.canonical(record, with_mask=True)[1])
            copied = copy.deepcopy(store.manifest)
            copied["records"][1]["split"] = "test"
            with self.assertRaises(ValueError):
                validate_manifest(copied)
            p, rr = annotation_subset(store.manifest["records"], 0.5)
            self.assertEqual(len(p), 2)
            self.assertEqual({r["patient_id"] for r in rr}, set(p))

    def test_native_spacing_axis_and_empty_prediction(self):
        a = np.zeros((8,8), bool); b = a.copy()
        a[2,2] = True; b[2,3] = True
        result = binary_metrics(a,b,[0.2,1.6])
        self.assertAlmostEqual(result["hd95"], 0.2)
        perfect = binary_metrics(a,a,[0.2,1.6])
        self.assertEqual(perfect["dice"],1)
        self.assertEqual(perfect["hd95"],0)
        fail = binary_metrics(np.zeros_like(a),a,[0.2,1.6])
        self.assertEqual(fail["dice"],0)
        self.assertGreater(fail["hd95"],0)
        self.assertIsNone(binary_metrics(a,np.zeros_like(a))["dice"])

    def test_letterbox_restores_native_coordinates(self):
        mask = np.zeros((31,67),np.int64)
        mask[8:23,15:51] = 1
        _, y, meta = letterbox(mask.astype(np.float32),mask,128)
        logits = torch.nn.functional.one_hot(y,4).permute(2,0,1).float()[None]*10
        restored = native_logits(logits,meta)[0].argmax(0).numpy()
        self.assertEqual(restored.shape,mask.shape)
        self.assertGreater(binary_metrics(restored==1,mask==1)["dice"],0.96)

    def test_patient_weighting_not_frame_weighting(self):
        base = {"class_id":1,"distance_unit":"pixels","iou":1.0,"hd95":0.0,"assd":0.0,
                "empty_prediction":0.0,"missing_reference":0}
        rows = [{**base,"patient_id":"long","dice":1.0} for _ in range(100)]
        rows.append({**base,"patient_id":"short","dice":0.0})
        result, _, _ = aggregate_case_metrics(rows)
        self.assertEqual(result["dice"],0.5)
        ci = paired_bootstrap({"a":1.,"b":0.}, {"a":0.5,"b":0.5}, repetitions=50)
        self.assertEqual(ci["mean_difference"],0)

    def test_calibration_perfect_and_patient_equal(self):
        labels = np.zeros((3,3),np.int64)
        probs = np.zeros((2,3,3)); probs[0] = 1
        row = probability_metrics(probs,labels)
        self.assertEqual(row["nll"],0)
        row["patient_id"] = "p1"
        bad = probability_metrics(probs,1-labels)
        bad["patient_id"] = "p2"
        result = aggregate_calibration([row]*50+[bad])
        self.assertAlmostEqual(result["ece"],0.5)

    def test_unsafe_archive_rejected(self):
        import zipfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"bad.zip"
            with zipfile.ZipFile(p,"w") as z:
                z.writestr("../escape.txt","bad")
            with self.assertRaises(ValueError):
                safe_extract(p,Path(d)/"out")

    def test_real_format_medical_io_and_official_splits(self):
        try:
            import SimpleITK as sitk
        except ImportError:
            self.skipTest("SimpleITK absent")
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"raw"; root.mkdir()
            splits={"train":["patient0001","patient0002"],"val":["patient0003"],"test":["patient0004"]}
            filenames={"train":"subgroup_training.txt","val":"subgroup_validation.txt","test":"subgroup_testing.txt"}
            for k,ids in splits.items():
                (root/filenames[k]).write_text("\n".join(ids))
            for patient in sum(splits.values(),[]):
                folder=root/patient; folder.mkdir()
                for view in ["2CH","4CH"]:
                    for phase in ["ED","ES"]:
                        mask=np.zeros((1,32,48),np.uint8)
                        mask[:,8:22,10:35]=2; mask[:,10:20,13:32]=1; mask[:,2:7,15:30]=3
                        image=(mask*50+10).astype(np.uint8)
                        for data,suffix in [(image,""),(mask,"_gt")]:
                            im=sitk.GetImageFromArray(data)
                            im.SetSpacing((0.2,0.7,1))
                            sitk.WriteImage(im,str(folder/f"{patient}_{view}_{phase}{suffix}.nii.gz"))
            cfg={"label_map_confirmed":True,"intermediate_frames_per_sequence":0}
            manifest=prepare_camus(root,Path(d)/"cache",cfg)
            store=RecordStore(manifest,64)
            self.assertEqual(store.manifest["counts"]["train"]["patients"],2)
            self.assertTrue(np.allclose(store.manifest["records"][0]["spacing_xy"],[0.2,0.7]))
            before=read_json(manifest)
            prepare_camus(root,Path(d)/"cache",{**cfg,"external_target":"myocardium"})
            self.assertEqual(before,read_json(manifest))
            (root/'.camus_download_cache').mkdir()
            write_json(root/'.camus_download_cache'/'progress.json',{'completed':3})
            write_json(root/'.camus_download_state.json',{'status':'complete'})
            prepare_camus(root,Path(d)/'cache',cfg)
            self.assertEqual(before,read_json(manifest))

    def test_resume_and_end_to_end_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            manifest=make_synthetic(d/"data")
            cfg=tiny_cfg()
            full=train(manifest,d/"full","aces",cfg,17,0.5,False)
            from aces import engine
            original=engine.atomic_torch_save
            interrupted=False
            def save_then_interrupt(obj,path):
                nonlocal interrupted
                original(obj,path)
                if Path(path).name=="last.pt" and obj["epoch"]==0 and obj["next_step"]==1 and not interrupted:
                    interrupted=True
                    raise KeyboardInterrupt
            with patch("aces.engine.atomic_torch_save",side_effect=save_then_interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    train(manifest,d/"resumed","aces",cfg,17,0.5,False)
            resumed=train(manifest,d/"resumed","aces",cfg,17,0.5,False)
            a,b=load_checkpoint(full)["model"],load_checkpoint(resumed)["model"]
            self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))
            temp=fit_temperature(resumed,manifest,d/"temperature.json",False,pixels_per_patient=64)
            result=evaluate(resumed,manifest,d/"resumed"/"test",temperature=temp,require_cuda=False,save_predictions=False)
            self.assertTrue(0 <= result["dice"] <= 1)
            self.assertTrue((d/"resumed"/"test"/"per_patient.csv").exists())
            report=export_report(d,d/"report")
            self.assertEqual(report["empirical_run_groups"],0)
            changed=cfg.copy(); changed["learning_rate"]=0.1
            with self.assertRaises(ValueError):
                train(manifest,d/"resumed","aces",changed,17,0.5,False)

    def test_external_mapping_and_pixel_units(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            image=np.zeros((20,30),np.float32); image[5:15,8:22]=100
            mask=(image>0).astype(np.uint8)
            np.save(d/"image.npy",image); np.save(d/"mask.npy",mask)
            p=d/"external.csv"
            p.write_text("patient_id,recording_id,image,mask,confirmed,foreground_value\np1,r1,image.npy,mask.npy,true,1\n")
            manifest=prepare_external(p,d/"cache","myocardium")
            store=RecordStore(manifest,64)
            self.assertIsNone(store.manifest["records"][0]["spacing_xy"])
            self.assertEqual(store.manifest["external_target"],"myocardium")
            with self.assertRaises(ValueError):
                prepare_external(p,d/"bad",None)

    def test_positive_unsupervised_anatomical_filter(self):
        with tempfile.TemporaryDirectory() as d:
            store=RecordStore(make_synthetic(d),64)
            _,rr=annotation_subset(store.manifest["records"],0.5)
            prior=AnatomicalPrior.fit(rr,store.canonical)
            x,mask,meta=store.canonical(rr[0])
            class Oracle(torch.nn.Module):
                def forward(self,x,view):
                    return {"logits":torch.nn.functional.one_hot(mask,4).permute(2,0,1).float()[None].to(x.device)*20}
            strong,pseudo,weights,score=pseudo_targets(Oracle(),x[None],torch.tensor([meta["view_id"]]),prior,
                                                      "student_t",confidence=0,score_threshold=0)
            self.assertGreater(weights.sum().item(),0)
            logits=torch.randn(1,4,64,64,requires_grad=True)
            unsupervised_loss(logits,pseudo,weights).backward()
            self.assertGreater(logits.grad.abs().sum().item(),0)

    def test_verified_external_inference_and_latency(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            manifest=make_synthetic(d/"data")
            cfg=tiny_cfg(); cfg["warmup_epochs"]=1
            checkpoint=train(manifest,d/"model","mean_teacher",cfg,17,0.5,False)
            record=read_json(manifest)["records"][-1]
            with np.load(d/"data"/record["path"]) as z:
                np.save(d/"image.npy",z["image"])
                np.save(d/"mask.npy",(z["mask"]==2).astype(np.uint8))
            csvpath=d/"external.csv"
            csvpath.write_text("patient_id,image,mask,confirmed\nexternalp,image.npy,mask.npy,true\n")
            external=prepare_external(csvpath,d/"external","myocardium")
            # The fixture is explicitly software-only, even in the external test.
            m=read_json(external); m["synthetic"]=True; write_json(external,m)
            summary=evaluate(checkpoint,external,d/"external_result",split="external",require_cuda=False,save_predictions=True)
            self.assertEqual(summary["distance_unit"],"pixels")
            self.assertTrue((d/"external_result"/"overlays").exists())
            benchmark(checkpoint,manifest,d/"latency.json",require_cuda=False,warmup=1,repetitions=2)
            self.assertGreater(read_json(d/"latency.json")["median_ms"],0)

    def test_three_seed_reporting_and_mismatched_manifest_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            for method in ["aces","unet"]:
                for seed in [17,29,43]:
                    target=d/f"{method}_{seed}"/"test"; target.mkdir(parents=True)
                    # Explicit unit-test fixture, not an actual trained clinical observation.
                    result={"synthetic":False,"method":method,"seed":seed,"label_fraction":0.25,
                            "split":"test","patients":2,"distance_unit":"mm","dataset_signature":"unit_test_fixture",
                            "dice":0.8 if method=="aces" else 0.7,"iou":0.5,"hd95":1.,"assd":0.2,"empty_prediction":0.}
                    write_json(target/"summary.json",result)
                    (target/"per_patient.csv").write_text("patient_id,dice\na,"+str(result["dice"])+"\nb,"+str(result["dice"])+"\n")
            report=export_report(d,d/"report")
            self.assertEqual(len(report["paired_comparisons"]),1)
            self.assertAlmostEqual(report["paired_comparisons"][0]["mean_difference"],0.1)
            f=d/"unet_17"/"test"/"summary.json"
            r=read_json(f);r["dataset_signature"]="wrong";write_json(f,r)
            with self.assertRaises(ValueError):
                export_report(d,d/"report")

    def test_video_and_mat_explicit_frame_mapping(self):
        try:
            import cv2
            from scipy.io import savemat
        except ImportError:
            self.skipTest("Optional video reader unavailable")
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            video=d/"echo.avi"
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"MJPG"),25,(24,16))
            if not writer.isOpened():
                self.skipTest("No MJPG encoder for fixture")
            for value in [30,100]:
                writer.write(np.full((16,24,3),value,np.uint8))
            writer.release()
            a,_=read_array(video,frame=1)
            self.assertEqual(a.shape,(16,24))
            self.assertGreater(a.mean(),80)
            masks=np.zeros((16,24,2),np.uint8);masks[3:10,5:15,1]=1
            savemat(d/"mask.mat",{"verified_mask":masks})
            b,_=read_array(d/"mask.mat",frame=1,mat_key="verified_mask",frame_axis=2)
            self.assertTrue(np.array_equal(b,masks[:,:,1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
