"""CPU-only candidate generators with private truth and exhaustive admission checks."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import secrets
from pathlib import Path
import numpy as np

from .contracts import Task,FAMILIES
from . import networks,programs
from .patches import diff_patch,materialize
from .serialization import canonical_json,save_weights,load_weights,validate_weights,encode_weights,decode_weights,weights_digest


def _write(path:Path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")


def _banks(seed:int):
    x=programs.domain_inputs();order=np.random.default_rng(seed).permutation(len(x))
    # Banks are frozen at generation, never chosen from queries a solver actually made.
    return x,order[:64],order[64:]


def _shift(node:dict):
    node=json.loads(json.dumps(node))
    if node["op"]=="mod":node["offset"]=(node["offset"]+1)%3
    elif node["op"]=="threshold":node["cuts"]=[v-1 for v in node["cuts"]]
    else:
        node["threshold"]-=1;node["then"]=_shift(node["then"]);node["else"]=_shift(node["else"])
    return node


def _pair(program:dict):
    other=json.loads(json.dumps(program));other["readout"]={"op":"mod","modulus":3,"offset":(len(programs.canonical(program))%3)}
    return [program,other]


def _interventions(arch:dict):
    # Distinct practice/final compositions; no arbitrary patch endpoint in forecasting.
    practice=[{"id":"p0","patch":[{"op":"scale","tensor":"q","factor":.9}]},
              {"id":"p1","patch":[{"op":"scale","tensor":"v","factor":.5}]},
              {"id":"p2","patch":[{"op":"zero","tensor":"w1","axis":0,"indices":[0,1,2]}]}]
    forecast=[{"id":"f0","patch":[{"op":"scale","tensor":"q","factor":1}]},
              {"id":"f1","patch":[{"op":"scale","tensor":"o","factor":.67}]},
              {"id":"f2","patch":[{"op":"zero","tensor":"w1","axis":0,"indices":[3,4,5]}]},
              {"id":"f3","patch":[{"op":"scale","tensor":"o","factor":.67},{"op":"zero","tensor":"w1","axis":0,"indices":[0,1,2]}]}]
    return practice,forecast


def _fault_hypotheses():
    sites=(1,6,11)
    choices=[(),(1,),(6,),(11,),(1,6),(1,11),(6,11)]
    return [{"id":f"h{i}","patch":[{"op":"scale","tensor":"w1","axis":1,"indices":list(indices),"factor":-1}] if indices else []} for i,indices in enumerate(choices)]


def _witnesses(arch:dict,faulty:dict,clean:dict,x:np.ndarray):
    truth=networks.predictions(arch,clean,x)
    predictions=[networks.predictions(arch,materialize(arch,faulty,{"patch":h["patch"]}),x) for h in _fault_hypotheses()]
    # Each viable hypothesis must have the clean complete-domain behavior; all alternatives get a witness.
    bad=[p for p in predictions if not np.array_equal(p,truth)]
    chosen=[]
    while bad:
        counts=np.asarray([sum(not np.array_equal(p[i],truth[i]) for p in bad) for i in range(len(x))])
        index=int(np.argmax(counts))
        if not counts[index]:raise ValueError("Fault witnesses not identifiable")
        chosen.append(index);bad=[p for p in bad if np.array_equal(p[index],truth[index])]
    normative=[{"input":x[i].tolist(),"expected":truth[i].tolist(),"domain_index":i} for i in chosen]
    viable=[i for i,p in enumerate(predictions) if all(np.array_equal(p[j],truth[j]) for j in chosen)]
    if not viable or any(not np.array_equal(predictions[i],truth) for i in viable):raise ValueError("Ambiguous fault class")
    return normative,{"hypotheses":len(predictions),"witnesses":len(chosen),"viable_equivalent_hypotheses":len(viable),"complete_domain_identifiable":True}


def _make(root:Path,seed:int,split:str,family:str,index:int,program:dict|None,stratum:str="constructed",label_index:int|None=None):
    rng=np.random.default_rng(seed)
    ident=hashlib.sha256(f"{seed}:{split}:{family}:{index}:{stratum}".encode()).hexdigest()[:16]
    directory=root/split/ident;directory.mkdir(parents=True,exist_ok=False)
    x,practice,final=_banks(seed%1000003)
    hint="mechanism" if family in ("coupled_repair","loop_repair") else "neutral"
    public={"domain":{"symbols":[0,1,2],"pad":3,"maximum_length":5,"count":len(x)},
            "practice_size":len(practice),"program_candidates":programs.candidates(),
            "answer_schema":{"type":"program"},
            "prompt_information":{"hint_level":hint,"diagnosis_scope":"fault_family" if family=="coupled_repair" else "looping_mechanism" if family=="loop_repair" else "task_specification","source":"benchmark_authored","optimization":"none","history":[]},
            "observation_policy":{"practice_only":True,"activations":True,"final_bank_fixed":True},
            "grammar_scope":"position-weighted symbol recoding, modular/threshold readouts, nested scalar conditionals"}
    private={"generation_seed":seed,"practice_indices":practice.tolist(),"final_indices":final.tolist(),"reference_access_conditions":[],"reference_attestation":"oracle program compiler proves representability; hidden gold program does not attest information-matched A/B/C feasibility"}
    before=None;training=None
    if family in ("rule_recovery","causal_forecast","edit_effects","compositional_synthesis"):
        declared=_pair(program) if family=="causal_forecast" else [program]
        arch=networks.attention_architecture(3*len(declared))
        clean=networks.compile_program(declared,arch)
        if stratum=="trained":
            clean=networks.fit_readout(arch,clean,np.column_stack([programs.evaluate_program(p,x) for p in declared]),x,rng,declared)
            training={"method":"supervised_least_squares","scope":"readout_on_constructed_attention_and_hinge_scaffold","training_examples":len(x),"complete_domain_teacher_agreement":True,"synthetic_aggregate_supervision":21,"independent_network_id":ident,"feature_seed":seed,"population_independence":"fresh_continuous_features_and_fit"}
        clean=networks.reparameterize(clean,rng)
        weights={k:v.copy() for k,v in clean.items()}
        private["program"]=program;private["label"]=programs.canonical(program)
        expected=np.column_stack([programs.evaluate_program(p,x) for p in declared])
        if not np.array_equal(networks.predictions(arch,clean,x),expected):raise ValueError("Compiler admission failure")
        if family=="rule_recovery":
            instruction="Describe this network's sequence-to-symbol behavior using the public program grammar. Submit {program: expression}."
            private["reference_submission"]={"program":program}
        elif family=="causal_forecast":
            p,f=_interventions(arch);public["practice_interventions"]=p;public["forecast_interventions"]=f
            public["declared_programs"]=declared;public["answer_schema"]={"type":"effects","units":"fraction of all 363 domain inputs with changed argmax, separately for each head","estimand":"exhaustive_domain_intervention_heldout","absolute_tolerance":.01}
            instruction="Predict changed-output fractions for each unqueried forecast intervention, compared with the supplied original network, over all 363 public-domain inputs and separately for both output heads. Final intervention IDs cannot be queried. Submit {effects:{id:[head0,head1]}}. Practice probes accept only practice IDs."
            effects={item["id"]:np.mean(networks.predictions(arch,materialize(arch,weights,{"patch":item["patch"]}),x)!=expected,axis=0).tolist() for item in f}
            private["reference_submission"]={"effects":effects};private["reference_access_conditions"]=[]
            private["reference_attestation"]="trusted exhaustive effect oracle only; no actual-permitted-information feasibility attestation"
        elif family=="edit_effects":
            before={k:v.copy() for k,v in weights.items()}
            shifted=json.loads(json.dumps(program));shifted["readout"]=_shift(program["readout"])
            if index%4==0:shifted=program
            else:weights["b0"]+=weights["w0"][:,0]
            if not np.array_equal(networks.predictions(arch,weights,x)[:,0],programs.evaluate_program(shifted,x)):raise ValueError("Internal edit not representable")
            public["answer_schema"]={"type":"before_after_programs"}
            instruction="Infer both sequence rules from before and after weights. Submit {before_program:expression,after_program:expression}. The evaluator derives changed-input sets and output consequences."
            private["after_program"]=shifted;private["reference_submission"]={"before_program":program,"after_program":shifted}
            expected=networks.predictions(arch,weights,x)
        else:
            public["specification"]=program;public["answer_schema"]={"type":"numeric_patch"}
            instruction="Construct numeric tensors for the supplied complete program specification in this fixed attention architecture, starting from zero. Submit {patch:[numeric operations]}."
            weights=networks.zeros(arch);private["reference_submission"]=diff_patch(weights,clean)
            private["reference_access_conditions"]=["B","C"];private["reference_attestation"]="analytic constructor consumes the fully public specification and emits numeric tensors without target execution; no unaided model certificate"
    elif family=="coupled_repair":
        arch=networks.dense_architecture();clean,training=networks.train_dense(rng)
        stratum="trained";weights={k:v.copy() for k,v in clean.items()}
        fault=() if index%4==0 else ((1,6),(1,11),(6,11))[(index-1)%3] if split=="heldout" else ((1,),(6,),(11,))[(index-1)%3]
        if fault:weights["w1"][:,list(fault)]*=-1
        normative,certificate=_witnesses(arch,weights,clean,x)
        witness_indices=[item.pop("domain_index") for item in normative]
        practice=np.asarray(witness_indices+[int(i) for i in practice if i not in witness_indices],dtype=int)
        practice=practice[:64]
        final=np.asarray([i for i in range(len(x)) if i not in set(practice.tolist())],dtype=int)
        for i,item in enumerate(normative):item["practice_index"]=i
        public.pop("program_candidates");public["fault_hypotheses"]=_fault_hypotheses();public["normative_examples"]=normative
        public["fault_class"]="One declared single/pair sign-inversion hypothesis (or identity) affected outgoing projections at shared sites 1,6,11; clean checkpoints are possible. Both heads must preserve the checkpoint-specific clean behavior."
        public["answer_schema"]={"type":"numeric_patch"}
        instruction="Restore the coupled computation under the declared finite sign-fault class. Normative examples are desired answers, not observed faulty outputs. Submit a numeric patch; retain both heads' learned behavior."
        private.update({"fault":fault,"identifiability_certificate":certificate,"reference_submission":diff_patch(weights,clean),"reference_access_conditions":["C"],"reference_attestation":"finite inverse-hypothesis execution against normative witnesses establishes C feasibility only; A/B feasibility requires independent attestation"})
        expected=networks.predictions(arch,clean,x)
    elif family=="loop_repair":
        arch=networks.loop_architecture();trigger=index%3;clean=networks.compile_loop(trigger,False)
        weights=networks.compile_loop(trigger,index%4!=0)
        public.pop("program_candidates");public["task_goal"]="Copy every nonpadding input token in order, then emit EOS token 3, including long repeated inputs."
        public["practice_size"]=len(practice);public["horizon"]=64;public["answer_schema"]={"type":"numeric_patch"}
        instruction="Mechanism hint: this recurrent model may have a repetition/halting attractor. Repair correct copying and completion while preserving valid long repetitive outputs. Submit a numeric weight patch. This is a diagnosis-supplied mechanism control."
        private.update({"reference_submission":diff_patch(weights,clean),"reference_access_conditions":["C"],"reference_attestation":"analytic constructed recurrent control with explicit loop mechanism hint","trigger":trigger})
        expected=np.zeros((len(x),1),dtype=int)
    else:raise ValueError("Unknown family")
    public["activation_scalars_per_example"]=arch["activation_scalars_per_example"]
    private["practice_indices"]=practice.tolist();private["final_indices"]=final.tolist()
    private["training"]=training
    if training is not None:training["feature_digest"]=weights_digest(clean)
    private["clean_control"]=bool(family in ("coupled_repair","loop_repair") and index%4==0)
    private["reference_weights_digest"]=weights_digest(clean)
    # Semantic parents are shared by independent parameterizations, preventing pseudoreplication.
    semantic=programs.canonical(program) if program is not None else f"{family}-mechanism-{split}-{index}"
    parent=hashlib.sha256(f"{family}:{semantic}:{seed if split=='baseline_train' or program is None else 'paired'}".encode()).hexdigest()[:16]
    mechanism=hashlib.sha256(f"{family}:{semantic}".encode()).hexdigest()[:16] if program else ("paired_shared_sign_fault" if split=="heldout" else "single_shared_sign_fault") if family=="coupled_repair" else f"eos_attractor_trigger_{index%3}"
    task=Task(ident,parent,mechanism,split,family,FAMILIES[family],stratum,"medium_4096",arch,instruction,public,directory,conditions=("C",) if family=="loop_repair" else ("A","B","C"))
    private["reference_submission"]=private["reference_submission"]
    save_weights(directory/"target.npz",weights);save_weights(directory/"clean.npz",clean)
    if before is not None:save_weights(directory/"before.npz",before)
    np.savez_compressed(directory/"evaluation.npz",inputs=x,expected=expected,practice=practice,final=final)
    _write(directory/"private.json",private);_write(directory/"task.json",task.to_public_dict())
    return task


def generate_suite(root:Path,seed:int|None=None,dev_per_family:int=2,heldout_per_family:int=4,baseline_per_label:int=32)->dict:
    root=Path(root)
    if root.exists() and any(root.iterdir()):raise ValueError("Generation requires an empty output directory")
    if any(type(n) is not int or n<0 for n in (dev_per_family,heldout_per_family,baseline_per_label)):raise ValueError("Counts must be nonnegative integers")
    seed=secrets.randbits(63) if seed is None else seed
    if type(seed) is not int or seed<0:raise ValueError("Seed must be a nonnegative integer")
    root.mkdir(parents=True,exist_ok=True)
    tasks=[];failures=[];counter=0
    for split,count in (("dev",dev_per_family),("heldout",heldout_per_family)):
        pool=programs.candidates(split)
        if count>len(pool):raise ValueError("Requested more independent programs than the frozen grammar split provides")
        for family in FAMILIES:
            for i in range(count):
                p=pool[i] if family not in ("coupled_repair","loop_repair") else None
                strata=("constructed","trained") if family in ("rule_recovery","causal_forecast","edit_effects") else ("trained",) if family=="coupled_repair" else ("constructed",)
                for stratum in strata:
                    counter+=1
                    try:tasks.append(_make(root,seed+counter*7919,split,family,i,p,stratum))
                    except ValueError as error:
                        failures.append({"split":split,"family":family,"index":i,"stratum":stratum,"reason":str(error)})
                        _write(root/"failed_admissions.json",failures)
                        raise
    for label,p in enumerate(programs.candidates()):
        for i in range(baseline_per_label):
            counter+=1
            tasks.append(_make(root,seed+counter*7919,"baseline_train","rule_recovery",i,p,"trained",label))
    summary={"benchmark_version":"0.2.0","status":"candidate_not_admitted","seed":seed,"tasks":len(tasks),"parents":len({t.parent_problem_id for t in tasks if t.split!="baseline_train"}),"by_split":{s:sum(t.split==s for t in tasks) for s in ("dev","heldout","baseline_train")},"baseline_per_label":baseline_per_label,"failed_admissions":failures,"training_scope":"attention trained stratum fits readout only on constructed features; coupled dense factors freshly trained","website":"deferred","paid_calls":0}
    _write(root/"suite.json",summary)
    return summary


def load_tasks(root:Path,split:str="dev")->list[Task]:
    if split not in ("dev","heldout","baseline_train","all"):raise ValueError("Unknown split")
    paths=Path(root).glob("*/*/task.json") if split=="all" else (Path(root)/split).glob("*/task.json")
    tasks=[Task.from_dict(json.loads(p.read_text()),p.parent) for p in sorted(paths)]
    if not tasks:raise ValueError(f"No {split} tasks in suite")
    return tasks


def generate_composed_suite(root:Path,seed:int|None=None,dev_per_family:int=4,
                            heldout_per_family:int=8,baseline_per_label:int=32)->dict:
    """Fresh composed candidates; old scalar/repair targets stay explicit controls."""
    from . import challenges
    root=Path(root)
    if root.exists() and any(root.iterdir()):raise ValueError("Generation requires an empty output directory")
    if any(type(n) is not int or n<0 for n in (dev_per_family,heldout_per_family,baseline_per_label)):
        raise ValueError("Counts must be nonnegative integers")
    if dev_per_family>4 or heldout_per_family>8:raise ValueError("Control split supports at most4development/8heldout parents")
    seed=secrets.randbits(63) if seed is None else seed
    if type(seed) is not int or seed<0:raise ValueError("Seed must be a nonnegative integer")
    root.mkdir(parents=True,exist_ok=True)
    tasks=[];counter=0;seen=set();semantic_splits={}
    def composed(split,family,index):
        nonlocal counter
        for retry in range(100):
            counter+=1
            sample_seed=seed+counter*7919
            bundle=challenges.make_challenge(sample_seed,family,split,index)
            # Related parameterizations never become separate mechanism claims;
            # reject duplicate complete function pairs across split/population.
            signature=hashlib.sha256(bundle["expected"].tobytes()+(networks.predictions(bundle["architecture"],bundle["before"],bundle["inputs"]).tobytes() if bundle["before"] is not None else b"")).hexdigest()
            key=(family,signature)
            functions=[hashlib.sha256(bundle["expected"].tobytes()).hexdigest()]
            if bundle["before"] is not None:
                functions.append(hashlib.sha256(networks.predictions(bundle["architecture"],bundle["before"],bundle["inputs"]).tobytes()).hexdigest())
            if key not in seen and all(f not in semantic_splits or semantic_splits[f]==split for f in functions):break
        else:raise ValueError("Could not generate a fresh semantic composition")
        seen.add(key)
        semantic_splits.update({f:split for f in functions})
        ident=hashlib.sha256(f"composed:{sample_seed}:{family}:{split}".encode()).hexdigest()[:16]
        directory=root/split/ident;directory.mkdir(parents=True)
        public=bundle["public"]
        public.update(generator_revision="composed-v2.1",evaluation_role="challenge_candidate" if split!="baseline_train" else "baseline_training")
        private=bundle["private"]
        if family=="compositional_synthesis":
            private["reference_submission"]={"patch":[{"op":"replace","tensor":k,"values":v.tolist()} for k,v in sorted(bundle["clean"].items())]}
        private.update(generation_seed=sample_seed,reference_access_conditions=[],
                       reference_attestation="recomputed public numeric derivation or complete-semantics capacity witness; see feasibility records")
        task=Task(ident,hashlib.sha256(f"{family}:{signature}".encode()).hexdigest()[:16],
                  f"{family}:{signature[:16]}",split,family,FAMILIES[family],bundle["target_stratum"],
                  "microcircuit_4049",bundle["architecture"],bundle["instruction"],public,directory)
        save_weights(task.weights_path,bundle["weights"]);save_weights(directory/"clean.npz",bundle["clean"])
        if bundle["before"] is not None:save_weights(directory/"before.npz",bundle["before"])
        np.savez_compressed(directory/"evaluation.npz",inputs=bundle["inputs"],expected=bundle["expected"],practice=bundle["practice"],final=bundle["final"])
        _write(task.private_path,private);_write(directory/"task.json",task.to_public_dict())
        return task
    for split,count in (("dev",dev_per_family),("heldout",heldout_per_family)):
        for family in FAMILIES:
            for i in range(count):
                if family in ("rule_recovery","edit_effects","compositional_synthesis"):
                    tasks.append(composed(split,family,i))
                else:
                    strata=("constructed","trained") if family=="causal_forecast" else ("trained",) if family=="coupled_repair" else ("constructed",)
                    for stratum in strata:
                        counter+=1
                        p=programs.candidates(split)[i] if family=="causal_forecast" else None
                        task=_make(root,seed+counter*7919,split,family,i,p,stratum)
                        task.public.update(generator_revision="composed-v2.1",evaluation_role="diagnostic_sanity")
                        _write(task.directory/"task.json",task.to_public_dict());tasks.append(task)
    # Four independently sampled root-operation strata,32fresh functions each
    # by default, for both original and post-edit structured supervision.
    for family in ("rule_recovery","edit_effects"):
        for i in range(4*baseline_per_label):tasks.append(composed("baseline_train",family,i))
    from .certificates import feasibility_records
    for task in tasks:
        if task.split!="baseline_train":feasibility_records(task,persist=True)
    summary={"benchmark_version":"0.2.0","generator_revision":"composed-v2.1","status":"candidate_not_admitted",
             "seed":seed,"tasks":len(tasks),"parents":len({t.parent_problem_id for t in tasks if t.split!="baseline_train"}),
             "by_split":{s:sum(t.split==s for t in tasks) for s in ("dev","heldout","baseline_train")},
             "baseline_per_root_operator":baseline_per_label,"baseline_supervision":"open-output truth-table regression; original and edited programs represented",
             "training_scope":"new targets are constructed microcircuits; retained dense repair factors/readout controls are trained",
             "website":"deferred","paid_calls":0}
    _write(root/"suite.json",summary)
    return summary


def validate_suite(root:Path)->dict:
    from .scoring import score,reference_submission
    tasks=load_tasks(root,"all");failures=[];families={};strata={};feasible={};unattested=[];proof_records=[];edit_balance={};semantic_splits={}
    for task in tasks:
        try:
            private=json.loads(task.private_path.read_text())
            conditions=set(private.get("reference_access_conditions",[]))
            feasible[task.family]=conditions if task.family not in feasible else feasible[task.family]&conditions
            if set(task.conditions)-conditions:unattested.append({"task_id":task.id,"conditions":sorted(set(task.conditions)-conditions),"reason":private.get("reference_attestation", "missing reference access attestation")})
            w=load_weights(task.weights_path);validate_weights(task.architecture,w)
            restored=decode_weights(encode_weights(w))
            if any(not np.array_equal(w[k],restored[k]) for k in w):raise ValueError("Serialization mismatch")
            with np.load(task.directory/"evaluation.npz",allow_pickle=False) as data:
                x=data["inputs"];expected=data["expected"];p=data["practice"];f=data["final"]
                if len(set(p.tolist())&set(f.tolist())) or sorted(np.r_[p,f].tolist())!=list(range(len(x))):raise ValueError("Practice/final overlap or missing inputs")
                clean=load_weights(task.directory/"clean.npz")
                if task.architecture.get("kind")=="boolean_relu":
                    signatures=[hashlib.sha256(expected.tobytes()).hexdigest()]
                    if task.family=="edit_effects":
                        before=networks.predictions(task.architecture,load_weights(task.directory/"before.npz"),x)
                        signatures.append(hashlib.sha256(before.tobytes()).hexdigest())
                        counts=edit_balance.setdefault(task.split,{}).setdefault(private["program"]["op"],{True:0,False:0})
                        counts[bool(np.array_equal(before,expected))]+=1
                    for signature in signatures:
                        if signature in semantic_splits and semantic_splits[signature]!=task.split:
                            raise ValueError("Equivalent Boolean functions cross development/held-out/training split boundaries")
                        semantic_splits[signature]=task.split
                if task.family!="loop_repair" and not np.array_equal(networks.predictions(task.architecture,clean,x),expected if task.family!="edit_effects" else networks.predictions(task.architecture,load_weights(task.directory/"before.npz"),x)):
                    # Edit-effect expected belongs to after; clean is before by construction.
                    raise ValueError("Reference checkpoint ground truth mismatch")
                if task.family=="loop_repair":
                    roll=networks.rollout(task.architecture,clean,x)
                    truth=[[int(t) for t in row if t!=3]+[3] for row in x]
                    if roll["generated_tokens"]!=truth:raise ValueError("Loop reference fails valid long outputs")
            if task.family=="causal_forecast":
                practice=task.public["practice_interventions"];forecast=task.public["forecast_interventions"]
                if set(p["id"] for p in practice)&set(p["id"] for p in forecast):raise ValueError("Intervention IDs overlap across catalogs")
                practice_digests={weights_digest(materialize(task.architecture,w,{"patch":p["patch"]})) for p in practice}
                final_digests={weights_digest(materialize(task.architecture,w,{"patch":p["patch"]})) for p in forecast}
                if practice_digests&final_digests:raise ValueError("Equivalent numeric interventions overlap across catalogs")
            result=score(task,reference_submission(task))
            if not result.success:raise ValueError(f"Reference score failed: {result}")
            if score(task,{"malformed":True}).status!="invalid":raise ValueError("Malformed submission accepted")
            if task.public.get("generator_revision")=="composed-v2.1" and task.split!="baseline_train":
                from .certificates import feasibility_records
                proof_records.extend(feasibility_records(task))
            families[task.family]=families.get(task.family,0)+1;strata[task.target_stratum]=strata.get(task.target_stratum,0)+1
        except (ValueError,OSError,KeyError) as error:failures.append({"id":task.id,"reason":str(error)})
    for split,groups in edit_balance.items():
        if sum(sum(counts.values()) for counts in groups.values())>=8:
            if any(counts[True]!=counts[False] for counts in groups.values()):
                failures.append({"split":split,"reason":"Edit sham status is not balanced within each root-operator stratum"})
    if proof_records:
        verified={};unattested=[]
        for task in tasks:
            if task.split=="baseline_train":continue
            conditions={r["condition"] for r in proof_records if r["task_id"]==task.id}
            verified[task.family]=conditions if task.family not in verified else verified[task.family]&conditions
            if set(task.conditions)-conditions:unattested.append({"task_id":task.id,"conditions":sorted(set(task.conditions)-conditions),"reason":"no checked condition-specific public-information witness"})
        feasible=verified
    return {"status":"failed" if failures else "passed","valid":not failures,"tasks":len(tasks),"families":families,"strata":strata,"failures":failures,"feasible_conditions_by_family":{k:sorted(v) for k,v in feasible.items()},"feasibility_records":proof_records,"unattested_reference_conditions":unattested,"edit_control_balance":{split:{op:{"sham":counts[True],"changed":counts[False]} for op,counts in groups.items()} for split,groups in edit_balance.items()},"admission_status":"candidate_not_admitted","checks":["complete-domain references","disjoint fixed banks","float32 serialization","malformed artifacts","numeric intervention identity separation","loop legitimate-long controls","public-information numeric derivations","capacity-constrained complete-semantics witnesses","sham/changed balance within root-operator strata","cross-split full-function equivalence separation"],"cpu_only":True}
