"""Offline information/constructibility checks, distinct from solver scores.

The verifier may execute a target to check an already constructed witness. Such
verification is not feedback delivered to a solver and is recorded separately.
"""
from __future__ import annotations

import json
import time
import numpy as np

from .patches import materialize
from .serialization import canonical_json,load_weights,serialization_digest,weights_digest


def linear_fault_derivation(public, architecture, weights):
    """Finite public linear inequalities; no checkpoint runtime or hidden labels.

    This is an offline bounded mathematical derivation. Computing its numerical
    inequalities as a trial tool would emulate a forward pass, so it is not
    advertised as an eligible B software solver.
    """
    witnesses=public["normative_examples"]
    if not witnesses:raise ValueError("Missing public normative witnesses")
    candidates=[]
    for hypothesis in public["fault_hypotheses"]:
        edited=materialize(architecture,weights,{"patch":hypothesis["patch"]})
        coefficients=edited["w1"].astype(float)@edited["w0"].astype(float)
        offsets=edited["w1"].astype(float)@edited["b0"].astype(float)+edited["b1"]
        comparisons=[];consistent=True
        for example in witnesses:
            active=[4*j+int(v) for j,v in enumerate(example["input"])]
            for head,desired in enumerate(example["expected"]):
                desired=int(desired)
                for other in range(3):
                    if other==desired:continue
                    a,b=head*3+desired,head*3+other
                    terms=(coefficients[a,active]-coefficients[b,active]).tolist()
                    bias=float(offsets[a]-offsets[b]);margin=float(sum(terms)+bias)
                    passed=margin>0 if other<desired else margin>=0
                    comparisons.append({"head":head,"desired":desired,"other":other,"terms":terms,"bias":bias,"margin":margin,"passed":passed})
                    consistent &= passed
        candidates.append({"id":hypothesis["id"],"consistent":bool(consistent),"comparisons":comparisons,"patch":hypothesis["patch"]})
    viable=[c for c in candidates if c["consistent"]]
    if not viable:raise ValueError("Public normative inequalities have no consistent inverse fault")
    proof={"method":"public_linear_inequality_certificate","candidates":candidates,
           "source":"disclosed weights, finite fault class and normative desired answers only",
           "trial_solver_claim":False,"caveat":"Numerically running these input-dependent inequalities in B would be forward emulation; this offline certificate demonstrates information sufficiency, not eligible B tool execution."}
    return {"patch":viable[0]["patch"]},proof


def feasibility_records(task,*,persist=False):
    from . import networks
    from .scoring import score
    started=time.process_time();weights=load_weights(task.weights_path)
    kind=task.architecture.get("kind");method=None;proof={};conditions=[];extra={}
    if kind=="boolean_relu":
        from .challenges import decode_program
        if task.family=="compositional_synthesis":
            spec=task.public["specification"]
            inputs=np.asarray(spec["inputs"],int);outputs=np.asarray(spec["outputs"],int)
            clean=load_weights(task.directory/"clean.npz")
            if len(outputs)!=1024 or not np.array_equal(networks.predictions(task.architecture,clean,inputs)[:,0],outputs):
                raise ValueError("Capacity witness does not realize the complete public semantics")
            method="public_semantics_capacity_witness";conditions=list(task.conditions)
            proof={"public_semantics_digest":serialization_digest(spec),"constructive_weights_digest":weights_digest(clean),
                   "persistent_scalars":sum(v.size for v in clean.values()),"witness_source":"private offline construction; not a competing public constructor"}
            extra={"complete_public_semantics":True,"constructive_witness_verified":True,"witness_privileged":True}
        else:
            decoded=decode_program(weights,task.architecture)
            submission={"program":decoded["program"]};proof={"after":decoded["certificate"]}
            if task.family=="edit_effects":
                before=decode_program(load_weights(task.directory/"before.npz"),task.architecture)
                submission={"before_program":before["program"],"after_program":decoded["program"]};proof["before"]=before["certificate"]
            if not score(task,submission).success:raise ValueError("Weights-only numeric derivation disagrees with trusted evaluator")
            method="public_numeric_microcircuit_derivation";conditions=list(task.conditions)
            proof["submission"]=submission
    elif task.family=="causal_forecast":
        from .feasibility import symbolic_effects
        result=symbolic_effects(weights,task.architecture,task.public["forecast_interventions"])
        if not score(task,{"effects":result["effects"]}).success:raise ValueError("Static causal witness disagrees with target semantics")
        proof=result;method="public_symbolic_piecewise_measure";conditions=["B","C"]
    elif task.family=="coupled_repair":
        submission,proof=linear_fault_derivation(task.public,task.architecture,weights)
        if not score(task,submission).success:raise ValueError("Public inequality derivation does not repair the full retained behavior")
        method="public_linear_inequality_certificate";conditions=["A","B","C"]
        extra["trial_solver_claim"]=False
    elif task.family=="loop_repair":
        method="public_copy_specification_construction";conditions=["C"]
        proof={"complete_public_goal":task.public["task_goal"],"source":"analytic copying state machine; independently tested spec-only baseline"}
    if method is None:return []
    # A constructive answer and a finite derivation fit the existing response
    # surface. This is not an estimate of a model's hidden reasoning requirements.
    reference=json.loads(task.private_path.read_text())["reference_submission"]
    solution_bytes=len(canonical_json(reference).encode())
    if solution_bytes>task.limits.output_tokens:
        raise ValueError("Reference answer exceeds the conservative one-byte-per-token output bound")
    bound=len(canonical_json(proof).encode())
    if bound>task.limits.output_tokens*3:raise ValueError("Offline derivation exceeds conservative response-size bound")
    proofpath=task.directory/"feasibility.json"
    if persist:
        proofpath.write_text(json.dumps(proof,indent=2,allow_nan=False)+"\n")
    elif not proofpath.exists() or serialization_digest(json.loads(proofpath.read_text()))!=serialization_digest(proof):
        raise ValueError("Persisted feasibility derivation is absent or changed; validation never repairs artifacts")
    elapsed=time.process_time()-started
    return [{"task_id":task.id,"target_digest":weights_digest(weights),"condition":condition,
             "status":"passed","method":method,"public_information_only":True,
             "public_derivation_checked":True,"uses_target_execution":False,
             "offline_verification_executed_target":True,"analysis_calls":0,
             "cpu_seconds":elapsed,"response_bytes_bound":bound,
             "model_output_tokens_bound":solution_bytes,
             "evidence_digest":serialization_digest(proof),**extra} for condition in conditions]
