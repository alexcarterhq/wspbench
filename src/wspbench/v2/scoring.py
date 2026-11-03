"""Trusted final grading and practice-only target instrumentation."""
from __future__ import annotations
import json
import numpy as np
from .contracts import Task,Score
from . import networks,programs
from .patches import materialize
from .serialization import load_weights,weights_digest,validate_weights,canonical_json


def reference_submission(task:Task)->dict:
    return json.loads(task.private_path.read_text())["reference_submission"]


def _bank(task):
    with np.load(task.directory/"evaluation.npz",allow_pickle=False) as data:
        return {name:data[name].copy() for name in data.files}


def _agreement(predicted,expected):
    return float(np.mean(np.asarray(predicted)==np.asarray(expected)))


def _patch_metrics(before,after):
    a=np.concatenate([before[k].reshape(-1) for k in sorted(before)]).astype(np.float64)
    b=np.concatenate([after[k].reshape(-1) for k in sorted(after)]).astype(np.float64)
    return {"edit_fraction":float(np.mean(a!=b)),"edit_norm":float(np.linalg.norm(b-a)),"parameters":int(len(a))}


def _expected_tokens(inputs):
    return [[int(t) for t in row if t!=3]+[3] for row in inputs]


def _read_prediction(submission, inputs, *, program_key="program", table_key="outputs"):
    from .challenges import evaluate_program
    if set(submission) == {program_key}:
        return evaluate_program(submission[program_key], inputs)
    if set(submission) == {table_key}:
        values = np.asarray(submission[table_key])
        if values.shape != (len(inputs),) or values.dtype.kind not in "iu" or np.any((values < 0) | (values > 1)):
            raise ValueError("Boolean output table must contain exactly 1024 binary integers in public domain order")
        return values
    raise ValueError("Submit a Boolean program or a complete binary output table")


def score(task:Task,submission:dict)->Score:
    try:
        if not isinstance(submission,dict):raise ValueError("Submission must be an object")
        data=_bank(task);x=data["inputs"];final=data["final"];expected=data["expected"]
        target=load_weights(task.weights_path)
        metrics={}
        if task.family=="rule_recovery":
            if task.architecture.get("kind") == "boolean_relu":
                predicted = _read_prediction(submission, x)
            else:
                if set(submission)!={"program"}:raise ValueError("Expected program field only")
                predicted=programs.evaluate_program(submission["program"],x)
            wanted=networks.predictions(task.architecture,target,x)[:,0]
            metrics={"functional_agreement":_agreement(predicted[final],wanted[final]),"exhaustive_agreement":_agreement(predicted,wanted)}
            success=metrics["functional_agreement"]==1
        elif task.family=="edit_effects":
            boolean = task.architecture.get("kind") == "boolean_relu"
            if not boolean and set(submission)!={"before_program","after_program"}:raise ValueError("Expected before_program and after_program")
            before=networks.predictions(task.architecture,load_weights(task.directory/"before.npz"),x)[:,0]
            after=networks.predictions(task.architecture,target,x)[:,0]
            if boolean:
                if set(submission) == {"before_program", "after_program"}:
                    from .challenges import evaluate_program
                    p=evaluate_program(submission["before_program"],x);q=evaluate_program(submission["after_program"],x)
                elif set(submission) == {"before_outputs", "after_outputs"}:
                    p=_read_prediction({"outputs":submission["before_outputs"]},x)
                    q=_read_prediction({"outputs":submission["after_outputs"]},x)
                else:
                    raise ValueError("Expected before/after Boolean programs or complete binary output tables")
            else:
                p=programs.evaluate_program(submission["before_program"],x);q=programs.evaluate_program(submission["after_program"],x)
            changed=(before!=after)[final];claimed=(p!=q)[final]
            tp=int(np.sum(changed&claimed));fp=int(np.sum(~changed&claimed));fn=int(np.sum(changed&~claimed))
            precision=tp/(tp+fp) if tp+fp else float(not np.any(changed))
            recall=tp/(tp+fn) if tp+fn else 1.0
            metrics={"before_agreement":_agreement(p[final],before[final]),"after_agreement":_agreement(q[final],after[final]),"affected_precision":precision,"affected_recall":recall,"false_positive_rate":float(np.mean(claimed&~changed)),"consequence_agreement":_agreement(q[final][changed],after[final][changed]) if changed.any() else 1.0,"exhaustive_agreement":float(np.mean((p==before)&(q==after)))}
            success=metrics["before_agreement"]==metrics["after_agreement"]==1
        elif task.family=="causal_forecast":
            if set(submission)!={"effects"} or not isinstance(submission["effects"],dict):raise ValueError("Expected effects mapping")
            private=json.loads(task.private_path.read_text());truth=private["reference_submission"]["effects"]
            if set(submission["effects"])!=set(truth):raise ValueError("Forecast IDs must match the final catalog")
            a=np.asarray([submission["effects"][k] for k in sorted(truth)])
            b=np.asarray([truth[k] for k in sorted(truth)])
            if a.shape!=b.shape or a.dtype.kind not in "fiu" or not np.isfinite(a).all() or np.any((a<0)|(a>1)):raise ValueError("Effects must be bounded numeric head fractions")
            errors=np.abs(a-b);metrics={"effect_mae":float(errors.mean()),"effect_max_error":float(errors.max()),"effect_accuracy":float(np.mean(errors<=.01))}
            success=bool(np.all(errors<=.01))
        elif task.family in ("coupled_repair","compositional_synthesis"):
            edited=materialize(task.architecture,target,submission)
            predicted=networks.predictions(task.architecture,edited,x)
            initial=networks.predictions(task.architecture,target,x)
            unchanged=np.all(initial==expected,axis=1);affected=~unchanged
            target_accuracy=_agreement(predicted[final],expected[final])
            preserved=unchanged[final]
            retention=_agreement(predicted[final][preserved],expected[final][preserved]) if preserved.any() else 1.0
            metrics={"target_accuracy":target_accuracy,"retention_accuracy":retention,"retention_loss":1-retention,"exhaustive_agreement":_agreement(predicted,expected),"affected_accuracy":_agreement(predicted[final][affected[final]],expected[final][affected[final]]) if affected[final].any() else 1.0,"false_positive_change_rate":float(np.mean(np.any(predicted[final][preserved]!=expected[final][preserved],axis=1))) if preserved.any() else 0.0,**_patch_metrics(target,edited)}
            # Finite exact tasks admit exact references: do not silently weaken to95%.
            success=target_accuracy==1 and retention==1
        elif task.family=="loop_repair":
            edited=materialize(task.architecture,target,submission)
            truth=_expected_tokens(x[final])
            initial=networks.rollout(task.architecture,target,x[final]);after=networks.rollout(task.architecture,edited,x[final])
            correct=np.asarray([a==b for a,b in zip(after["generated_tokens"],truth)])
            prior=np.asarray([a==b for a,b in zip(initial["generated_tokens"],truth)])
            repeated=np.asarray([len(set(t[:-1]))==1 and len(t)>3 for t in truth])
            prematurely_stopped=[bool(c and len(a)<len(b)) for a,b,c in zip(after["generated_tokens"],truth,after["completed"])]
            metrics={"correct_completion":float(correct.mean()),"completion_rate":float(np.mean(after["completed"])),"retention_accuracy":float(correct[prior].mean()) if prior.any() else 1.0,"retention_loss":1-float(correct[prior].mean()) if prior.any() else 0.,"valid_repetition_accuracy":float(correct[repeated].mean()) if repeated.any() else 1.0,"truncation_rate":float(np.mean(prematurely_stopped)),"noncompletion_rate":1-float(np.mean(after["completed"])),**_patch_metrics(target,edited)}
            success=bool(correct.all())
        else:raise ValueError("Unknown family")
        return Score(bool(success),metrics)
    except (ValueError,KeyError,TypeError,IndexError,OverflowError) as error:
        return Score(False,{},"invalid",str(error))


def probe(task:Task,weights:dict,indices,observation:str="behavior",checkpoint:str="target",intervention_id:str|None=None)->dict:
    if observation not in ("behavior","activations"):raise ValueError("Unknown observation")
    if not isinstance(indices,list) or not indices or any(type(i) is not int or i<0 or i>=task.public["practice_size"] for i in indices):raise ValueError("Practice indices must be in the frozen practice bank")
    if checkpoint not in ("target","before","after"):raise ValueError("Unknown checkpoint")
    if task.family!="edit_effects" and checkpoint!="target":raise ValueError("Only edit-effects has before/after checkpoints")
    validate_weights(task.architecture,weights)
    bank=_bank(task);x=bank["inputs"][bank["practice"][indices]]
    accounting={"example_evaluations":len(indices),"activation_scalars":0,"episodes":0,"steps":0,"generated_tokens":0}
    result={"indices":indices,"inputs":x.tolist()}
    if task.family=="causal_forecast":
        if weights_digest(weights)!=weights_digest(load_weights(task.weights_path)):
            raise ValueError("Forecast probes require the unchanged true checkpoint")
        catalog={item["id"]:item for item in task.public["practice_interventions"]}
        if intervention_id not in catalog:raise ValueError("Only predeclared practice intervention IDs may be executed")
        baseline=networks.predictions(task.architecture,weights,x)
        weights=materialize(task.architecture,weights,{"patch":catalog[intervention_id]["patch"]})
        accounting["example_evaluations"]=2*len(indices)
        result["baseline_outputs"]=baseline.tolist()
    elif intervention_id is not None:raise ValueError("Intervention IDs are only defined for forecast tasks")
    if checkpoint=="before":weights=load_weights(task.directory/"before.npz")
    elif checkpoint=="after":weights=load_weights(task.weights_path)
    if task.family=="loop_repair":
        rollout=networks.rollout(task.architecture,weights,x,observation=="activations")
        result.update(rollout)
        truth=_expected_tokens(x)
        result["correct"]=[a==b for a,b in zip(rollout["generated_tokens"],truth)]
        result["expected_tokens"]=truth
        accounting.update(example_evaluations=0,episodes=len(x),steps=rollout["steps"],generated_tokens=rollout["steps"])
        if observation=="activations":accounting["activation_scalars"]=rollout["steps"]*24
    else:
        logits,activations=networks.forward(task.architecture,weights,x,True)
        outputs=logits.reshape(len(x),task.architecture["heads"],3).argmax(axis=2)
        result["outputs"]=outputs.tolist()
        if task.family in ("coupled_repair","compositional_synthesis"):
            result["expected"]=bank["expected"][bank["practice"][indices]].tolist()
        result["logits"]=logits.tolist()
        if task.family=="causal_forecast":result["change_rates"]=np.mean(outputs!=baseline,axis=0).tolist()
        if observation=="activations":
            result["activations"]={k:v.tolist() for k,v in activations.items()}
            accounting["activation_scalars"]=sum(v.size for v in activations.values())
    result["accounting"]=accounting
    return result
