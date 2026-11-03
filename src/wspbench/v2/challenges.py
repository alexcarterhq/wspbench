"""Procedural Boolean circuit challenges and a numeric, fixed ReLU runtime.

The compiler is a private construction witness for truth-table synthesis. Its
input AST is never needed by the target runtime. Read targets are deliberately
transparent constructed microcircuits: a weights-only decompiler is a legitimate
classical ceiling for B, rather than evidence of a difficult B interpretation task.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from itertools import product
import json

import numpy as np

from .serialization import canonical_json, validate_weights, weights_digest
from .patches import diff_patch

INPUTS = 10
WIDTH = 34
STAGES = 4
BINARY = ("and", "or", "xor", "nand", "nor")


def architecture() -> dict:
    shapes = {"w0": [WIDTH, INPUTS], "b0": [WIDTH]}
    for layer in range(1, STAGES):
        shapes[f"w{layer}"] = [WIDTH, WIDTH]
        shapes[f"b{layer}"] = [WIDTH]
    shapes.update(out=[3, WIDTH], bias=[3])
    return {"kind": "boolean_relu", "revision": "procedural-boolean-v1", "shapes": shapes,
            "heads": 1, "input_shape": [INPUTS], "output_shape": [1],
            "boundary_tensors": ["w0", "b0", "out", "bias"],
            "activation_scalars_per_example": STAGES * WIDTH,
            "forward_definition": "h=x; for i in range(4): h=relu(h@wi.T+bi); logits=h@out.T+bias; output=argmax(logits)",
            "constructed_layout": "Channels 0..9 retain inputs. Each consecutive triple after channel9 encodes one Boolean signal as h[j]+h[j+1]-h[j+2]. Unused channels are zero. This layout is intentionally available to algebraic comparators."}


def domain_inputs() -> np.ndarray:
    return np.asarray(list(product((0, 1), repeat=INPUTS)), dtype=np.int64)


def _children(node: dict) -> list[tuple[str, dict]]:
    op = node.get("op")
    if op in BINARY:
        return [("left", node["left"]), ("right", node["right"])]
    if op == "not":
        return [("arg", node["arg"])]
    if op == "mux":
        return [("condition", node["condition"]), ("then", node["then"]), ("else", node["else"])]
    if op == "majority":
        return [(str(i), n) for i, n in enumerate(node["args"])]
    return []


def validate_program(program: dict) -> None:
    count = 0
    def visit(node, depth):
        nonlocal count
        count += 1
        if not isinstance(node, dict) or count > 128 or depth > 6:
            raise ValueError("Boolean expression exceeds the declared grammar")
        op = node.get("op")
        expected = {"input": {"op", "index"}, "not": {"op", "arg"},
                    "mux": {"op", "condition", "then", "else"},
                    "majority": {"op", "args"}}
        keys = {"op", "left", "right"} if op in BINARY else expected.get(op)
        if keys is None or set(node) != keys:
            raise ValueError("Unknown or malformed Boolean expression")
        if op == "input":
            if type(node["index"]) is not int or not 0 <= node["index"] < INPUTS:
                raise ValueError("Input index outside public domain")
        elif op == "majority":
            if not isinstance(node["args"], list) or len(node["args"]) != 3:
                raise ValueError("Majority requires exactly three operands")
        for _, child in _children(node):
            visit(child, depth + 1)
    visit(program, 0)


def _depth(node: dict) -> int:
    children = _children(node)
    return 0 if not children else 1 + max(_depth(n) for _, n in children)


def evaluate_program(program: dict, inputs) -> np.ndarray:
    validate_program(program)
    x = np.asarray(inputs)
    if x.ndim != 2 or x.shape[1] != INPUTS or x.dtype.kind not in "iu" or np.any((x < 0) | (x > 1)):
        raise ValueError("Expected ten binary input values")
    def run(node):
        op = node["op"]
        if op == "input": return x[:, node["index"]].astype(bool)
        if op == "not": return ~run(node["arg"])
        if op == "mux": return np.where(run(node["condition"]), run(node["then"]), run(node["else"]))
        if op == "majority": return sum(run(n).astype(int) for n in node["args"]) >= 2
        a, b = run(node["left"]), run(node["right"])
        if op == "and": return a & b
        if op == "or": return a | b
        if op == "xor": return a ^ b
        if op == "nand": return ~(a & b)
        if op == "nor": return ~(a | b)
        raise ValueError("Unknown Boolean operation")
    return run(program).astype(np.int64)


def zeros(arch: dict | None = None) -> dict[str, np.ndarray]:
    return {name: np.zeros(shape, np.float32) for name, shape in (arch or architecture())["shapes"].items()}


def _gate_rows(op: str, operands: list[np.ndarray]) -> np.ndarray:
    """Three affine rows whose ReLUs combine as first+second-third."""
    z = np.zeros_like(operands[0])
    one = z.copy(); one[-1] = 1
    if op == "not": return np.stack((one-operands[0], z, z))
    if op == "mux":
        c, t, e = operands
        return np.stack((c+t-one, e-c, z))
    s = sum(operands)
    if op == "majority": return np.stack((s-one, z, s-2*one))
    if op == "and": return np.stack((s-one, z, z))
    if op == "or": return np.stack((s, z, s-one))
    if op == "xor": return np.stack((s, z, 2*s-2*one))
    if op == "nand": return np.stack((one, z, s-one))
    if op == "nor": return np.stack((one, s-one, s))
    raise ValueError("Unsupported Boolean gate")


def compile_program(program: dict, seed: int = 0) -> dict[str, np.ndarray]:
    """Construct an exact bounded circuit. ``seed`` is accepted for API stability.

    Canonical neuron grouping is deliberate. Randomized filenames or hidden
    compiler layouts must not substitute for interpretation difficulty.
    """
    del seed
    validate_program(program)
    if _depth(program) != STAGES:
        raise ValueError("Construction witness requires a four-stage expression")
    nodes: dict[int, list[tuple[tuple[str, ...], dict]]] = {k: [] for k in range(1, STAGES+1)}
    def collect(node, path):
        for field, child in _children(node): collect(child, path+(field,))
        depth = _depth(node)
        if depth: nodes[depth].append((path, node))
    collect(program, ())
    arch = architecture(); w = zeros(arch)
    representations: dict[tuple[str, ...], np.ndarray] = {}
    for layer in range(STAGES):
        n_in = INPUTS if layer == 0 else WIDTH
        for bit in range(INPUTS): w[f"w{layer}"][bit, bit] = 1
        current = {}
        if len(nodes[layer+1]) > 8: raise ValueError("Circuit exceeds eight gates at a stage")
        for slot, (path, node) in enumerate(nodes[layer+1]):
            operands = []
            for field, child in _children(node):
                affine = np.zeros(n_in+1, np.float32)
                if child["op"] == "input": affine[child["index"]] = 1
                else:
                    if _depth(child) != layer:
                        raise ValueError("Non-input gate must consume a previous-stage signal")
                    affine[:-1] = representations[path+(field,)]
                operands.append(affine)
            rows = _gate_rows(node["op"], operands)
            start = INPUTS+3*slot
            w[f"w{layer}"][start:start+3] = rows[:, :-1]
            w[f"b{layer}"][start:start+3] = rows[:, -1]
            signal = np.zeros(WIDTH, np.float32); signal[start:start+3] = (1, 1, -1)
            current[path] = signal
        representations = current
    signal = representations[()]
    w["out"][0] = -signal; w["bias"][0] = 1
    w["out"][1] = signal; w["bias"][2] = -1
    validate_weights(arch, w)
    return w


def forward(arch: dict, weights: dict, inputs, return_activations: bool = False):
    if arch.get("kind") != "boolean_relu": raise ValueError("Wrong Boolean architecture")
    validate_weights(arch, weights)
    x = np.asarray(inputs)
    if x.ndim != 2 or x.shape[1] != INPUTS or x.dtype.kind not in "iu" or np.any((x < 0) | (x > 1)):
        raise ValueError("Expected ten binary input values")
    h = x.astype(np.float32)
    activations = {}
    for layer in range(STAGES):
        h = np.maximum(h @ weights[f"w{layer}"].T + weights[f"b{layer}"], 0)
        activations[f"hidden{layer}"] = h
    logits = h @ weights["out"].T + weights["bias"]
    return (logits, activations) if return_activations else logits


def predictions(arch: dict, weights: dict, inputs) -> np.ndarray:
    return forward(arch, weights, inputs).argmax(axis=1)[:, None]


def sample_program(seed: int, mechanism: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    mechanism = seed % 4 if mechanism is None else mechanism % 4
    x = domain_inputs()
    def tree(depth):
        if not depth: return {"op": "input", "index": int(rng.integers(INPUTS))}
        return {"op": str(rng.choice(BINARY)), "left": tree(depth-1), "right": tree(depth-1)}
    for _ in range(1000):
        left, right = tree(3), tree(3)
        bit = {"op": "input", "index": int(rng.integers(INPUTS))}
        if mechanism == 0: program = {"op": "xor", "left": left, "right": right}
        elif mechanism == 1: program = {"op": "mux", "condition": bit, "then": left, "else": right}
        elif mechanism == 2: program = {"op": "majority", "args": [left, right, bit]}
        else: program = {"op": "nand", "left": left, "right": right}
        y = evaluate_program(program, x)
        relevant = sum(np.any(y != evaluate_program(program, x ^ np.eye(INPUTS, dtype=np.int64)[j])) for j in range(INPUTS))
        if .1 <= y.mean() <= .9 and relevant >= 6: return program
    raise ValueError("Could not sample a nondegenerate procedural composition")


def _decode_gate(rows: np.ndarray, signals: list[dict]) -> dict:
    """Interpret exact affine gate rows; never execute a target or inspect its source AST."""
    if not np.array_equal(rows, np.round(rows)):
        raise ValueError("Nonintegral gate coefficients are outside the attested construction")
    rows = np.round(rows)
    zero = np.zeros(rows.shape[1]); one = zero.copy(); one[-1] = 1
    def literal_sum(row, count):
        coeff = row[:-1]
        if not np.all(np.isclose(coeff, np.round(coeff))) or np.any(coeff < 0): return None
        indices = [i for i, n in enumerate(coeff.astype(int)) for _ in range(n)]
        return indices if len(indices) == count else None
    for op in BINARY:
        if op in ("and", "or", "xor"):
            base = rows[0].copy(); base[-1] = 0
        else:
            base = rows[2].copy(); base[-1] = 0
        args = literal_sum(base, 2)
        if args is not None:
            operands = []
            for k in args:
                vector=zero.copy();vector[k]=1;operands.append(vector)
            if np.allclose(rows, _gate_rows(op, operands), atol=1e-5):
                return {"op":op,"left":deepcopy(signals[args[0]]),"right":deepcopy(signals[args[1]])}
    base=rows[0].copy();base[-1]=0
    args=literal_sum(base,3)
    if args is not None:
        vectors=[]
        for k in args:
            v=zero.copy();v[k]=1;vectors.append(v)
        if np.allclose(rows,_gate_rows("majority",vectors),atol=1e-5):return {"op":"majority","args":[deepcopy(signals[k]) for k in args]}
    # Mux normally exposes its control as the negative coefficient in row1.
    negatives=np.flatnonzero(rows[1,:-1] < -.5)
    positives=np.flatnonzero(rows[1,:-1] > .5)
    if len(negatives)==len(positives)==1:
        c,e=int(negatives[0]),int(positives[0]);then=rows[0].copy();then[-1]=0;then[c]-=1
        terms=literal_sum(then,1)
        if terms:
            t=terms[0]
            vectors=[]
            for k in (c,t,e):v=zero.copy();v[k]=1;vectors.append(v)
            if np.allclose(rows,_gate_rows("mux",vectors),atol=1e-5):return {"op":"mux","condition":deepcopy(signals[c]),"then":deepcopy(signals[t]),"else":deepcopy(signals[e])}
    raise ValueError("Weights are outside the attested public microcircuit grammar")


def derive_program(weights: dict) -> dict:
    """An information-matched weights-only derivation for the constructed stratum.

    This is an exact classical B comparator. It is not evidence that an unaided
    model can perform the derivation within its token allowance.
    """
    validate_weights(architecture(), weights)
    signals = [{"op":"input","index":i} for i in range(INPUTS)]
    vectors = np.eye(INPUTS)
    proof=[]
    for layer in range(STAGES):
        matrix,bias=weights[f"w{layer}"],weights[f"b{layer}"]
        expected=np.zeros((INPUTS,matrix.shape[1]));expected[:,:INPUTS]=np.eye(INPUTS)
        if not np.array_equal(matrix[:INPUTS],expected) or np.any(bias[:INPUTS]):
            raise ValueError("Input identity channels do not match the public construction")
        # Signal vectors have disjoint supports with squared norms1or3.
        # Exact contraction avoids accepting a small unsupported coefficient
        # merely because a general pseudoinverse passes an approximate check.
        coefficients=(matrix@vectors.T)/np.sum(vectors*vectors,axis=1)
        if not np.array_equal(coefficients@vectors,matrix):
            raise ValueError("Gate rows contain an unsupported hidden mixture")
        current=[{"op":"input","index":i} for i in range(INPUTS)]
        next_vectors=[np.eye(WIDTH)[i] for i in range(INPUTS)]
        for slot in range(8):
            start=INPUTS+3*slot
            if not np.any(matrix[start:start+3]) and not np.any(bias[start:start+3]):continue
            rows=np.column_stack([coefficients[start:start+3],bias[start:start+3]])
            gate=_decode_gate(rows,signals)
            current.append(gate)
            v=np.zeros(WIDTH);v[start:start+3]=(1,1,-1);next_vectors.append(v)
            proof.append({"stage":layer,"slot":slot,"op":gate["op"]})
        signals=current;vectors=np.asarray(next_vectors)
    candidates=[i for i,v in enumerate(vectors) if np.array_equal(weights["out"][1],v)]
    if len(candidates)!=1 or not np.array_equal(weights["out"][0],-vectors[candidates[0]]) or not np.array_equal(weights["bias"],np.array([1,0,-1],np.float32)) or np.any(weights["out"][2]):
        raise ValueError("Readout is outside the public construction")
    program=signals[candidates[0]]
    return {"program":program,"certificate":{"source":"serialized_weights_only","target_executions":0,"hidden_program_access":False,"steps":proof,"claim":"information sufficiency and a checkable numeric derivation; not an unaided model performance certificate"}}


def decode_program(weights: dict, architecture: dict | None = None) -> dict:
    if architecture is not None and architecture.get("kind") != "boolean_relu":
        raise ValueError("Wrong public architecture for this numeric derivation")
    return derive_program(weights)


def make_challenge(seed: int, family: str, split: str = "dev", index: int = 0) -> dict:
    if family not in ("rule_recovery","edit_effects","compositional_synthesis"):
        raise ValueError("This generator covers Read rule/effect and Write synthesis")
    program=sample_program(seed,index%4);arch=architecture();clean=compile_program(program)
    x=domain_inputs();expected=evaluate_program(program,x)[:,None]
    if not np.array_equal(predictions(arch,clean,x),expected):raise ValueError("Construction witness failed")
    rng=np.random.default_rng(seed+1);order=rng.permutation(len(x));practice=order[:256];final=order[256:]
    public={"challenge_revision":"procedural-boolean-v1","domain":{"symbols":[0,1],"features":INPUTS,"count":len(x)},"practice_size":len(practice),"activation_scalars_per_example":STAGES*WIDTH,
            "grammar":{"inputs":INPUTS,"operators":[*BINARY,"not","mux","majority"],"maximum_depth":6,"maximum_nodes":128,"definition":"and/or/xor/nand/nor use left,right; not uses arg; mux uses condition,then,else; majority uses a three-element args list; input uses index"},
            "answer_schema":{"type":"boolean_program_or_complete_outputs","output_count":len(x),"ordering":"lexicographic binary tuples, input0 most significant"},
            "prompt_information":{"hint_level":"neutral","diagnosis_scope":"task_specification","source":"benchmark_authored","optimization":"none","history":[]}}
    private={"program":program,"label":canonical_json(program),"reference_access_conditions":["B","C"],"reference_attestation":"public fixed microcircuit layout admits a verified serialized-weights-only symbolic derivation; this does not certify unaided model performance","reference_submission":{"program":program}}
    before=None;weights=clean
    instruction="Recover this ten-input Boolean function from its neural weights. Submit {program: Boolean expression} or {outputs:1024 binary values in lexicographic input order}. No finite answer catalog is supplied."
    if family=="edit_effects":
        before=clean;after=deepcopy(program)
        paths=[]
        def collect(node,path):
            if node["op"] in BINARY and 2<=_depth(node)<=3:paths.append(path)
            for field,child in _children(node):collect(child,path+(field,))
        collect(after,())
        def at(node,path):
            for field in path:node=node["args"][int(field)] if node["op"]=="majority" else node[field]
            return node
        # Eight-instance blocks cross root operator with clean/changed status.
        sham=(index//4+index%4)%2==0
        if not sham:
            for path in rng.permutation(len(paths)):
                candidate=deepcopy(program);node=at(candidate,paths[int(path)])
                original=node["op"];node["op"]={"and":"or","or":"and","xor":"nand","nand":"xor","nor":"or"}[original]
                changed=np.mean(evaluate_program(candidate,x)!=expected[:,0])
                if changed>=.05:
                    after=candidate;break
            else:raise ValueError("No nontrivial internal edit available")
        weights=compile_program(after)
        if any(not np.array_equal(before[k],weights[k]) for k in arch["boundary_tensors"]):raise ValueError("Edit changed boundary tensors")
        private.update(after_program=after,reference_submission={"before_program":program,"after_program":after})
        public["answer_schema"]={"type":"before_after_boolean_programs_or_complete_outputs","output_count":len(x),"ordering":"lexicographic binary tuples, input0 most significant"}
        instruction="Infer the Boolean functions before and after an internal weight edit. Submit {before_program:expression,after_program:expression} or {before_outputs:1024 bits,after_outputs:1024 bits}. Include function-preserving possibilities."
        expected=evaluate_program(after,x)[:,None]
    elif family=="compositional_synthesis":
        weights=zeros(arch)
        public["specification"]={"format":"truth_table","inputs":x.tolist(),"outputs":expected[:,0].tolist(),"domain_order":"lexicographic_binary_10"}
        public["answer_schema"]={"type":"numeric_patch","constraints":"fixed architecture; every persistent scalar counts toward4049"}
        private.update(reference_submission=diff_patch(weights,clean),reference_access_conditions=[],reference_attestation="complete public extensional specification plus private compact construction witness proves representability; no gold AST is supplied to a competing constructor")
        instruction="Construct this complete public Boolean truth table in the fixed four-stage ReLU network, starting from zero. Submit only {patch:[numeric operations]}. The truth table is the full desired behavior; the compact generating circuit is not supplied. Exact logic minimization and analytic circuit construction are permitted; target-parameter fitting, gradients and learned runtime predictors are prohibited."
    identity=hashlib.sha256(expected.tobytes()).hexdigest()[:16]
    private["reference_program"]=program
    private["reference_weights_digest"]=weights_digest(clean)
    return {"architecture":arch,"weights":weights,"before":before,"clean":clean,"public":public,"private":private,"instruction":instruction,"expected":expected,"inputs":x,"practice":practice,"final":final,"mechanism_family_id":f"boolean_function_{identity}","target_stratum":"constructed_microcircuit"}
