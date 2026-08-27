# PLE: Progressive Layered Extraction

Tags: multi-task, architecture, seesaw

Source: Tang et al., "Progressive Layered Extraction (PLE): A Novel Multi-Task Learning
Model for Personalized Recommendations", RecSys 2020 (best paper).

## What it is

MMoE's successor. It makes the sharing structure explicit rather than learned: each task
gets its own private experts *and* access to a pool of shared experts. A task's gate mixes
its own experts with the shared ones, but never with another task's private experts.

    tower_input_k = gate_k( private_experts_k + shared_experts )

Stacking several such layers gives the "progressive" part — deeper layers extract
increasingly task-specific representations.

## Why it helps a ranking metric

PLE's paper documents that MMoE's gates, in practice, often fail to separate conflicting
tasks — they converge to near-uniform mixtures and the seesaw persists. Hard-coding the
private/shared split guarantees each task keeps parameters no other task can perturb,
which is a structural fix rather than a learned one.

For a ranking metric where one label is scored and several are auxiliary, this means the
scored task retains dedicated capacity that a dense auxiliary label cannot overwrite.

## How to implement

Same caveat as MMoE, more so: this is the most architecturally involved multi-task option
in this corpus. The single-layer version ("CGC", Customized Gate Control) is the one to
implement if any — it is PLE without the stacking, and captures most of the benefit.

    for each task k:
        experts_k = private_experts[k] + shared_experts
        g = softmax(W_k @ x)                 # over len(experts_k)
        tower_input_k = sum_i g[i] * experts_k[i](x)

Suggested sizing for a small dataset: 1 private expert per task and 1-2 shared experts.
The paper's larger configurations assume industrial data volumes.

## When it will not help

- Same conditions as MMoE: no measured seesaw means no problem to solve.
- The parameter count grows with (tasks x private experts) and small datasets will not
  support it.
- It is a refinement of MMoE, which is a refinement of auxiliary losses. Skipping to the
  end of that chain without measuring the intermediate steps means you will not know
  which part produced any gain you observe.
