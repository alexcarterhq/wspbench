"""Frozen NumPy target architectures. Inference depends on numeric tensors only."""
from __future__ import annotations
import numpy as np
from .serialization import load_weights, save_weights, validate_weights
from .programs import readout_values, domain_inputs


def attention_architecture(outputs: int = 3) -> dict:
    if outputs not in (3, 6):
        raise ValueError("Only one or two three-class heads")
    shapes = {"e":[5,12], "p":[6,12], "q":[12,12], "k":[12,12], "v":[12,12], "o":[12,12],
              "w0":[32,12], "b0":[32], "w1":[outputs,32], "b1":[outputs]}
    return {"kind":"attention_pool", "shapes":shapes, "heads":outputs//3,
            "boundary_tensors":["e", "p", "w1", "b1"],
            "input_shape":[5], "output_shape":[outputs//3], "activation_scalars_per_example":205,
            "forward_definition":"x=e[concat(input,[4])]+p; a=softmax((x@q.T)[-1]@(x@k.T)[:5].T/sqrt(12)); z=(a@(x@v.T)[:5])@o.T; logits=relu(z@w0.T+b0)@w1.T+b1"}


def dense_architecture() -> dict:
    return {"kind":"dense_shared", "shapes":{"w0":[16,20], "b0":[16], "w1":[6,16], "b1":[6]},
            "heads":2, "boundary_tensors":["w0","b0","w1","b1"], "input_shape":[5],"output_shape":[2],
            "activation_scalars_per_example":22,
            "forward_definition":"x=flatten(one_hot(input,4)); h=x@w0.T+b0; logits=h@w1.T+b1"}


def loop_architecture() -> dict:
    return {"kind":"relu_recurrent", "shapes":{"encoder":[24,20], "encoder_bias":[24], "r":[24,24], "rb":[24], "out":[4,24], "ob":[4]},
            "heads":1,"boundary_tensors":["encoder","encoder_bias","out","ob"],"input_shape":[5],"output_shape":[64],
            "activation_scalars_per_example":64*28,"horizon":64,
            "forward_definition":"h=relu(one_hot(input,4).flatten()@encoder.T+encoder_bias); emit argmax(out@h+ob), stop on token3; h=relu(r@h+rb)"}


def zeros(architecture: dict) -> dict:
    return {name:np.zeros(shape,dtype=np.float32) for name,shape in architecture["shapes"].items()}


def forward(architecture: dict, weights: dict, inputs, return_activations: bool = False):
    if architecture.get("kind") == "boolean_relu":
        from .challenges import forward as boolean_forward
        return boolean_forward(architecture, weights, inputs, return_activations)
    validate_weights(architecture, weights)
    x = np.asarray(inputs)
    if x.ndim != 2 or x.shape[1] != 5 or x.dtype.kind not in "iu" or np.any((x<0)|(x>3)):
        raise ValueError("Expected padded input tokens")
    w = weights
    if architecture["kind"] == "attention_pool":
        ids=np.concatenate([x,np.full((len(x),1),4)],axis=1)
        hidden=w["e"][ids]+w["p"][None,:,:]
        q=hidden@w["q"].T; k=hidden@w["k"].T; v=hidden@w["v"].T
        scores=np.einsum("nd,nkd->nk",q[:,-1],k[:,:5])/np.sqrt(12)
        scores-=scores.max(axis=1,keepdims=True)
        a=np.exp(scores);a/=a.sum(axis=1,keepdims=True)
        z=np.einsum("nk,nkd->nd",a,v[:,:5])@w["o"].T
        h=np.maximum(z@w["w0"].T+w["b0"],0)
        logits=h@w["w1"].T+w["b1"]
        activations={"hidden":h,"pooled":z,"attention":a}
    elif architecture["kind"] == "dense_shared":
        encoded=np.eye(4,dtype=np.float32)[x].reshape(len(x),20)
        h=encoded@w["w0"].T+w["b0"]
        logits=h@w["w1"].T+w["b1"]
        activations={"hidden":h}
    else:
        raise ValueError("Use rollout for recurrent architecture")
    if return_activations:
        return logits,activations
    return logits


def predictions(architecture: dict, weights: dict, inputs) -> np.ndarray:
    if architecture.get("kind") == "boolean_relu":
        from .challenges import predictions as boolean_predictions
        return boolean_predictions(architecture, weights, inputs)
    logits=forward(architecture,weights,inputs)
    return logits.reshape(len(logits),architecture["heads"],3).argmax(axis=2)


def compile_program(program: dict | list, architecture: dict | None = None) -> dict:
    programs=program if isinstance(program,list) else [program]
    arch=architecture or attention_architecture(3*len(programs))
    if len(programs)!=arch["heads"] or any(p["codes"]!=programs[0]["codes"] or p["positions"]!=programs[0]["positions"] for p in programs):
        raise ValueError("Heads must share the declared aggregation")
    w=zeros(arch); p=programs[0]
    for i in range(5):w["e"][i,i]=1
    w["e"][:,11]=1
    for i in range(6):w["p"][i,5+i]=1
    w["q"][0,11]=np.sqrt(12)
    for i,a in enumerate(p["positions"]):w["k"][0,5+i]=np.log(a)
    w["v"][0,:3]=p["codes"]
    w["o"][0,0]=sum(p["positions"])
    knots=np.arange(-1,31)
    w["w0"][:,0]=1;w["b0"]=-knots.astype(np.float32)
    for head,program in enumerate(programs):
        for t, label in enumerate(readout_values(program,np.arange(21))):
            for delta,coeff in ((-1,1),(0,-2),(1,1)):
                w["w1"][3*head+label,t+delta+1]+=coeff
    validate_weights(arch,w)
    return w


def reparameterize(weights: dict, rng: np.random.Generator) -> dict:
    """Positive hidden rescaling/permutation; no semantic changes or extra buffers."""
    w={k:v.copy() for k,v in weights.items()}
    if "w0" in w:
        permutation=rng.permutation(len(w["b0"]))
        scale=rng.uniform(.55,1.8,len(permutation)).astype(np.float32)
        w["w0"]=w["w0"][permutation]*scale[:,None]
        w["b0"]=w["b0"][permutation]*scale
        w["w1"]=w["w1"][:,permutation]/scale[None,:]
    return w


def fit_readout(architecture: dict, weights: dict, labels: np.ndarray, inputs: np.ndarray, rng: np.random.Generator | None = None, teacher_programs: list | None = None) -> dict:
    """Genuine supervised least-squares fitting, on a constructed fixed feature scaffold."""
    w={k:v.copy() for k,v in weights.items()}
    if rng is not None:
        # Independently sampled feature knots change off-domain functions, rather than
        # merely permuting/rescaling the checkpoint of another training example.
        knots=np.arange(-1,31)+rng.uniform(.1,.4,32)
        gain=rng.uniform(.65,1.5,32)
        w["w0"][:,0]=gain.astype(np.float32)
        w["b0"]=(-knots*gain).astype(np.float32)
    _,a=forward(architecture,w,inputs,True)
    h=np.concatenate([a["hidden"],np.ones((len(inputs),1))],axis=1).astype(np.float64)
    y=np.eye(3)[labels].reshape(len(inputs),-1)
    if teacher_programs is not None:
        pooled=np.zeros((21,12));pooled[:,0]=np.arange(21)
        synthetic=np.maximum(pooled@w["w0"].T+w["b0"],0)
        h=np.concatenate([h,np.column_stack([synthetic,np.ones(21)])])
        synthetic_labels=np.column_stack([readout_values(p,np.arange(21)) for p in teacher_programs])
        y=np.concatenate([y,np.eye(3)[synthetic_labels].reshape(21,-1)])
    solution=np.linalg.lstsq(h,y,rcond=1e-7)[0]
    w["w1"]=solution[:-1].T.astype(np.float32);w["b1"]=solution[-1].astype(np.float32)
    if not np.array_equal(predictions(architecture,w,inputs),labels):
        raise ValueError("Trained readout did not meet exhaustive teacher agreement")
    return w


def train_dense(rng: np.random.Generator) -> tuple[dict,dict]:
    """Fresh low-rank shared linear network, fitted from random initialization to dense teacher logits."""
    arch=dense_architecture();x=domain_inputs();encoded=np.eye(4)[x].reshape(len(x),20)
    # Both heads depend on every shared channel and on checkpoint-specific teacher behavior.
    teacher=rng.normal(0,.8,(20,6));teacher_bias=rng.normal(0,.3,6)
    y=encoded@teacher+teacher_bias
    w=zeros(arch);w["w0"]=rng.normal(0,.3,(16,20)).astype(np.float32)
    w["b0"]=rng.normal(0,.1,16).astype(np.float32)
    # Alternating least squares is optimization of two dense coupled factors from random initialization.
    xaug=np.column_stack([encoded,np.ones(len(x))])
    second=rng.normal(0,.3,(17,6))
    for _ in range(4):
        first=np.linalg.lstsq(xaug,y@np.linalg.pinv(second),rcond=1e-8)[0]
        hidden=xaug@first
        second=np.linalg.lstsq(hidden,y,rcond=1e-8)[0]
    # A rank-six teacher needs only six effective dimensions; all sixteen channels are mixed densely.
    w["w0"]=first[:20,:16].T.astype(np.float32);w["b0"]=first[20,:16].astype(np.float32)
    residual=y-(encoded@w["w0"].T+w["b0"])@second[:16]
    # Fit the frozen network output exactly after dropping the homogeneous extra coordinate.
    hidden=encoded@w["w0"].T+w["b0"]
    solution=np.linalg.lstsq(np.column_stack([hidden,np.ones(len(x))]),y,rcond=1e-8)[0]
    w["w1"]=solution[:16].T.astype(np.float32);w["b1"]=solution[16].astype(np.float32)
    if not np.array_equal(predictions(arch,w,x),y.reshape(len(x),2,3).argmax(axis=2)):
        raise ValueError("Dense training admission failed")
    return w,{"method":"alternating_least_squares", "steps":5,"training_examples":len(x),"teacher":"fresh_dense_linear_logits", "scope":"all_dense_factors", "maximum_logit_error":float(np.max(np.abs(forward(arch,w,x)-y)))}


def compile_loop(trigger: int=0, faulty: bool=True) -> dict:
    arch=loop_architecture();w=zeros(arch)
    w["encoder"][:20,:20]=np.eye(20)
    w["encoder"][21,trigger]=1
    w["r"][:16,4:20]=np.eye(16)
    w["rb"][19]=1
    w["r"][21,21]=1
    w["r"][20,20]=2;w["r"][20,7]=2;w["r"][20,21]=2;w["rb"][20]=-3
    w["out"][:4,:4]=np.eye(4)
    if faulty:w["out"][0,20]=4;w["out"][3,20]=-4
    return w


def rollout(architecture: dict, weights: dict, inputs, return_activations=False) -> dict:
    validate_weights(architecture,weights)
    if architecture["kind"]!="relu_recurrent":raise ValueError("Not recurrent")
    rows=[]; hidden=[]; completed=[]
    for tokens in np.asarray(inputs,dtype=int):
        if tokens.shape!=(5,) or np.any((tokens<0)|(tokens>3)):raise ValueError("Bad tokens")
        h=np.maximum(weights["encoder"]@np.eye(4,dtype=np.float32)[tokens].reshape(-1)+weights["encoder_bias"],0)
        sequence=[];states=[]
        for _ in range(architecture["horizon"]):
            states.append(h.copy());token=int(np.argmax(weights["out"]@h+weights["ob"]))
            sequence.append(token)
            if token==3:break
            h=np.maximum(weights["r"]@h+weights["rb"],0)
            # Overflow is an observable failure, bounded before float overflow can occur.
            h=np.clip(h,0,1e12)
        rows.append(sequence);completed.append(sequence[-1]==3);hidden.append(states)
    result={"generated_tokens":rows,"completed":completed,"steps":sum(map(len,rows))}
    if return_activations:result["activations"]={"hidden":[np.asarray(h).tolist() for h in hidden]}
    return result
