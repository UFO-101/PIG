#%%

"""Currently a notebook so that I can develop the 16 Heads tests fast"""

import math
from IPython import get_ipython

if get_ipython() is not None:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

import argparse
import gc
from copy import deepcopy

import torch
import wandb
import tqdm

from transformer_lens import HookedTransformer 
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.acdc_utils import (
    cleanup,
    ct,
    kl_divergence,
    make_nd_dict,
    shuffle_tensor,
)

from acdc.TLACDCEdge import (
    Edge,
    EdgeType,
    TorchIndex,
)

from acdc.acdc_utils import reset_network
from acdc.docstring.utils import get_all_docstring_things
from acdc.greaterthan.utils import get_all_greaterthan_things
from acdc.induction.utils import (
    get_all_induction_things,
    get_good_induction_candidates,
    get_mask_repeat_candidates,
    get_validation_data,
)
from acdc.ioi.utils import get_all_ioi_things
from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.TLACDCInterpNode import TLACDCInterpNode, heads_to_nodes_to_mask
from acdc.tracr_task.utils import get_all_tracr_things
from subnetwork_probing.train import correspondence_from_mask, iterative_correspondence_from_mask
from notebooks.emacs_plotly_render import set_plotly_renderer

from subnetwork_probing.transformer_lens.transformer_lens.HookedTransformer import HookedTransformer as SPHookedTransformer
from subnetwork_probing.transformer_lens.transformer_lens.HookedTransformerConfig import HookedTransformerConfig as SPHookedTransformerConfig
from subnetwork_probing.train import do_random_resample_caching, do_zero_caching
from subnetwork_probing.transformer_lens.transformer_lens.hook_points import MaskedHookPoint

# set_plotly_renderer("emacs")

#%%

parser = argparse.ArgumentParser(description="Used to launch ACDC runs. Only task and threshold are required")
parser.add_argument('--task', type=str, required=True, choices=['ioi', 'docstring', 'induction', 'tracr-reverse', 'tracr-proportion', 'greaterthan'], help='Choose a task from the available options: ioi, docstring, induction, tracr-reverse, tracr-proportion, greaterthan')
parser.add_argument('--zero-ablation', action='store_true', help='Use zero ablation')
parser.add_argument('--wandb-entity', type=str, required=False, default="remix_school-of-rock", help='Value for WANDB_ENTITY_NAME')
parser.add_argument('--wandb-group', type=str, required=False, default="default", help='Value for WANDB_GROUP_NAME')
parser.add_argument('--wandb-project', type=str, required=False, default="acdc", help='Value for WANDB_PROJECT_NAME')
parser.add_argument('--wandb-run-name', type=str, required=False, default=None, help='Value for WANDB_RUN_NAME')
parser.add_argument("--wandb-dir", type=str, default="/tmp/wandb")
parser.add_argument("--wandb-mode", type=str, default="online")
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--reset-network', type=int, default=0, help="Whether to reset the network we're operating on before running interp on it")
parser.add_argument('--metric', type=str, default="kl_div", help="Which metric to use for the experiment")
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--torch-num-threads', type=int, default=0, help="How many threads to use for torch (0=all)")

# for now, force the args to be the same as the ones in the notebook, later make this a CLI tool
if get_ipython() is not None: # heheh get around this failing in notebooks
    args = parser.parse_args([line.strip() for line in r"""--task=docstring \
--wandb-group=pig \
--wandb-entity=josephmiller101\
--wandb-project=pig-not-normalized-1 \
--device=cpu \
--metric=docstring_metric \
--reset-network=0
""".split("\\\n")]) # so easy to copy and paste into terminal!!!
# --wandb-run-name=tracr-reverse-pig-1\

else:
    args = parser.parse_args()

torch.manual_seed(args.seed)

if args.torch_num_threads > 0:
    torch.set_num_threads(args.torch_num_threads)
torch.manual_seed(args.seed)

wandb.init(
    name=args.wandb_run_name,
    project=args.wandb_project,
    entity=args.wandb_entity,
    group=args.wandb_group,
    config=args,
    dir=args.wandb_dir,
    mode=args.wandb_mode,
)

#%%
print(" --------- TASK:", args.task)
if args.task == "ioi":
    num_examples = 100
    things = get_all_ioi_things(num_examples=num_examples, device=args.device, metric_name=args.metric)
elif args.task == "tracr-reverse":
    num_examples = 6
    things = get_all_tracr_things(task="reverse", metric_name=args.metric, num_examples=num_examples, device=args.device)
elif args.task == "tracr-proportion":
    num_examples = 50
    things = get_all_tracr_things(task="proportion", metric_name=args.metric, num_examples=num_examples, device=args.device)
elif args.task == "induction":
    num_examples = 5
    seq_len = 300
    # TODO initialize the `tl_model` with the right model
    things = get_all_induction_things(num_examples=num_examples, seq_len=seq_len, device=args.device, metric=args.metric)
elif args.task == "docstring":
    num_examples = 50
    seq_len = 41
    things = get_all_docstring_things(num_examples=num_examples, seq_len=seq_len, device=args.device,
                                                metric_name=args.metric, correct_incorrect_wandb=True)
elif args.task == "greaterthan":
    num_examples = 100
    things = get_all_greaterthan_things(num_examples=num_examples, metric_name=args.metric, device=args.device)
else:
    raise ValueError(f"Unknown task {args.task}")

# %% load the model into a Subnetwork-Probing model.
# We don't use the sixteen_heads=True argument any more, because we want to keep QKV separated.
# Deleted the 16H true argument altogether...


kwargs = dict(**things.tl_model.cfg.__dict__)

for extra_arg in [
    "use_split_qkv_input",
    "n_devices", # extra from new merge
    "gated_mlp",
]:
    if extra_arg in kwargs:
        del kwargs[extra_arg]

cfg = SPHookedTransformerConfig(**kwargs)
model = SPHookedTransformer(cfg, is_masked=True)
base_model = SPHookedTransformer(cfg, is_masked=True)
_acdc_model = things.tl_model
model.load_state_dict(_acdc_model.state_dict(), strict=False)
model = model.to(args.device)
base_model = base_model.to(args.device)

if args.reset_network:
    with torch.no_grad():
        reset_network(args.task, args.device, model)
        reset_network(args.task, args.device, _acdc_model)
        gc.collect()
        torch.cuda.empty_cache()


class SimpleMaskedHookPoint(MaskedHookPoint):
    def sample_mask(self, *args, **kwargs):
        # Directly return the scores instead of passing them through a sigmoid
        return self.mask_scores

for module in model.modules():
    if isinstance(module, MaskedHookPoint):
        module.__class__ = SimpleMaskedHookPoint

def replace_masked_hook_points(model):
    for n, c in model.named_children():
        if isinstance(c, MaskedHookPoint):
            setattr(model, n, SimpleMaskedHookPoint(mask_shape=c.mask_scores.shape, name=c.name, is_mlp=c.is_mlp).to(args.device))
        else:
            replace_masked_hook_points(c)
with torch.no_grad():
    replace_masked_hook_points(model)
model.freeze_weights()

# Set the masks to 1, so nothing is masked
with torch.no_grad():
    for n, p in model.named_parameters():
        if n.endswith("mask_scores"):
            p.fill_(1)

# Check that the model's outputs are the same
with torch.no_grad():
    expected = _acdc_model(things.validation_data).cpu()
    things.tl_model = None
    gc.collect()
    torch.cuda.empty_cache()

    actual = model(things.validation_data).cpu()
    gc.collect()
    torch.cuda.empty_cache()

    torch.testing.assert_allclose(
        actual, expected,
        atol=1e-3,
        rtol=1e-2,
    )


# %%

pig_scores = {n: None for n, c in model.named_modules() if isinstance(c, SimpleMaskedHookPoint)}

if model.cfg.attn_only == True or model.cfg.d_mlp == -1:
    # Attention-only model
    for k in list(pig_scores.keys()):
        if "mlp" in k:
            del pig_scores[k]

SAMPLES = 200
weights_to_attribute = {}
for n in pig_scores.keys():
    if "attn.hook" in n:
        weights_to_attribute[n] = n.replace("hook", "W")[:-1] + n[-1].upper()
    elif "hook_mlp_out" in n:
        weights_to_attribute[n] = n.replace("hook_mlp_out", "mlp.W_out")
    else:
        raise ValueError(n)

integrated_gradients = {n: torch.zeros_like(model.state_dict()[n]) for n in weights_to_attribute.values()}
prev_loss = None
prev_grads = None
model.train()
for n, p in list(model.named_parameters()):
    if n in weights_to_attribute.values():
        p.requires_grad = True
    else:
        p.requires_grad = False
# Run integrated gradients over the model weights between the base_model and _acdc_model
loss_diff_sum = 0
for i in range(0, SAMPLES + 1):
    model.zero_grad()
    new_weights = {}
    for n in weights_to_attribute.values():
        assert n.startswith("blocks")
        assert "attn" in n or "mlp" in n
        base_weight = base_model.state_dict()[n]
        acdc_weight = _acdc_model.state_dict()[n]
        # new_weights[n] = (1 - i / SAMPLES) * base_weight + (i / SAMPLES) * acdc_weight
        new_weights[n] = base_weight + (i / SAMPLES) * (acdc_weight - base_weight)
    for n, p in model.state_dict().items():
        new_weights.setdefault(n, p)
    # Load the new weights
    model.load_state_dict(new_weights, strict=True)
    # Run the model
    validation_output = model(things.validation_data)
    loss = things.validation_metric(validation_output, return_one_element=False).mean()
    # loss = validation_output.mean()
    loss.backward()

    if prev_loss is not None and prev_grads is not None:
        loss_diff = loss.item() - prev_loss
        loss_diff_sum += loss_diff
        prev_grads_sum = sum([torch.sum(p ** 2).item() for _, p in prev_grads.items()])
        for n, p in prev_grads.items():
            integrated_gradients[n] += (p ** 2 * loss_diff) / prev_grads_sum

    prev_loss = loss.item()
    prev_grads = {n: dict(model.named_parameters())[n].grad for n in weights_to_attribute.values()}

for pig_n, weight_n in weights_to_attribute.items():
    integrated_grad = integrated_gradients[weight_n]
    if "attn.hook" in pig_n:
        pig_scores[pig_n] = integrated_grad.sum(dim=tuple(range(1, integrated_grad.ndim)))
    elif "mlp_out" in pig_n:
        pig_scores[pig_n] = torch.tensor([integrated_grad.sum()])
    else:
        raise ValueError(pig_n)

new_weights = {}
for n in weights_to_attribute.values():
    new_weights[n] = base_model.state_dict()[n]
for n, p in model.state_dict().items():
    new_weights.setdefault(n, p)
model.load_state_dict(new_weights, strict=True)
base_output = model(things.validation_data)
base_loss = things.validation_metric(base_output, return_one_element=False)

new_weights = {}
for n in weights_to_attribute.values():
    new_weights[n] = _acdc_model.state_dict()[n]
for n, p in model.state_dict().items():
    new_weights.setdefault(n, p)
model.load_state_dict(new_weights, strict=True)
_acdc_output = model(things.validation_data)
_acdc_loss = things.validation_metric(_acdc_output, return_one_element=False)

# print("things.validation_data", things.validation_data)
# validation_output = model(things.validation_data)
# print("validation_output", validation_output)
# per_example_metric = things.validation_metric(validation_output, return_one_element=False)
# per_example_metric.retain_grad()
# assert per_example_metric.ndim == 1

# for n, c in model.named_modules():
#     if isinstance(c, SimpleMaskedHookPoint):
#         if c.mask_scores.grad is not None:
#             pig_scores[n] += c.mask_scores.grad.abs().detach()

# print("prune_scores after", pig_scores)

#%%

nodes_names_indices = []
for layer_i in range(model.cfg.n_layers):
    keys = [
        f"blocks.{layer_i}.attn.hook_{qkv}" for qkv in ["q", "k", "v"]
    ] + [f"blocks.{layer_i}.hook_mlp_out"]
    keys = [k for k in keys if k in pig_scores]

    layer_vector = torch.cat([pig_scores[k].flatten() for k in keys])
    norm = layer_vector.norm()

    # normalize by L2 of the layers
    # for k in keys:
    #     pig_scores[k] /= norm.clamp(min=1e-6)

    for qkv in ["q", "k", "v"]:
        for head_i in range(model.cfg.n_heads):
            name = f"blocks.{layer_i}.attn.hook_{qkv}"
            nodes = [TLACDCInterpNode(name, TorchIndex((None, None, head_i)), incoming_edge_type=EdgeType.ADDITION),
                     TLACDCInterpNode(f"blocks.{layer_i}.hook_{qkv}_input", TorchIndex((None, None, head_i)), incoming_edge_type=EdgeType.PLACEHOLDER)
                     ]
            nodes_names_indices.append((nodes, name, head_i))

    if model.cfg.attn_only == False and model.cfg.d_mlp != -1:
        name = f"blocks.{layer_i}.hook_mlp_out"
        mlp_nodes = [
            TLACDCInterpNode(name, TorchIndex([None]), incoming_edge_type=EdgeType.PLACEHOLDER),
            TLACDCInterpNode(f"blocks.{layer_i}.hook_mlp_in", TorchIndex([None]), incoming_edge_type=EdgeType.ADDITION),
        ]
        nodes_names_indices.append((mlp_nodes, name, slice(None)))


# sort by scores
nodes_names_indices.sort(key=lambda x: pig_scores[x[1]][x[2]].item(), reverse=True)

# %%
serializable_nodes_names_indices = [(list(map(str, nodes)), name, repr(idx), pig_scores[name][idx].item()) for nodes, name, idx in nodes_names_indices]
wandb.log({"nodes_names_indices": serializable_nodes_names_indices})

# %%

def test_metrics(logits, score):
    d = {"test_"+k: fn(logits).mean().item() for k, fn in things.test_metrics.items()}
    d["score"] = score
    return d

# Log metrics without ablating anything
logits = do_random_resample_caching(model, things.test_data)
wandb.log(test_metrics(logits, math.inf))

# %%

do_random_resample_caching(model, things.test_patch_data)
if args.zero_ablation:
    do_zero_caching(model)

import copy
nodes_to_mask = []
corr = None
head_parents = None
for nodes, hook_name, idx in tqdm.tqdm(nodes_names_indices):
    nodes_to_mask += nodes
    # COMPARE TO OLD!!!
    corr, head_parents = iterative_correspondence_from_mask(corr, head_parents, model, new_nodes=nodes, use_pos_embed=False, newv=False)
    for e in corr.all_edges().values():
        e.effect_size = 1.0

    score = pig_scores[hook_name][idx].item()

    # Delete this module
    done = False
    for n, c in model.named_modules():
        if n == hook_name:
            assert not done, f"Found {hook_name}[{idx}]twice"
            with torch.no_grad():
                c.mask_scores[idx] = 0
            done = True
    assert done, f"Could not find {hook_name}[{idx}]"

    to_log_dict = test_metrics(model(things.test_data), score)
    to_log_dict["number_of_edges"] = corr.count_no_edges()

    wandb.log(to_log_dict)

# %%

wandb.finish()

#%%

