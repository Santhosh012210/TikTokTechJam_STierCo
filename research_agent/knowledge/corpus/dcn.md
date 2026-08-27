# DCN and DCN-V2: Deep and Cross Network

Tags: architecture, feature-interaction, ctr

Source: Wang et al., "Deep & Cross Network for Ad Click Predictions", ADKDD 2017,
arXiv:1708.05123. DCN-V2: Wang et al., WWW 2021, arXiv:2008.13535

## What it is

A network with an explicit cross layer that computes bounded-degree feature interactions
in a structured way. Each cross layer applies:

    x_{l+1} = x_0 * (w_l^T x_l) + b_l + x_l          # DCN
    x_{l+1} = x_0 * (W_l x_l + b_l) + x_l            # DCN-V2, W is a matrix

After `l` layers the output contains interactions up to degree `l+1`. DCN-V2 replaces the
vector `w` with a matrix `W`, greatly increasing expressiveness; it is the version worth
implementing.

## Why it helps a ranking metric

Compared to DeepFM's MLP branch, the cross network learns interactions of *known,
bounded degree* rather than hoping an MLP discovers them. It is more parameter-efficient
than an MLP at the same interaction order, and in the DCN-V2 paper's comparisons it
outperformed MLP-based alternatives on several CTR benchmarks.

The same scepticism as DeepFM applies: this is an argument that higher-order interaction
is the bottleneck, which is an empirical claim about a specific dataset.

## How to implement

The cross layer is genuinely simple — a few lines forward, and the backward pass is
tractable by hand because the structure is explicit:

    # forward, DCN-V2
    xl_w = xl @ W + b
    out = x0 * xl_w + xl

    # backward
    d_xl_w = grad_out * x0
    dW = xl.T @ d_xl_w
    d_xl = d_xl_w @ W.T + grad_out

`x0` is the concatenated embedding vector of all fields for that row and stays fixed
across layers, which is what keeps the degree bounded and interpretable.

Start with 2 cross layers. The paper's ablations show returns diminish quickly beyond 3.

## When it will not help

Same conditions as DeepFM, and the same evidence check applies: if capacity ablations
show a flat metric across embedding dimensions, added interaction capacity is not the
binding constraint.

Additionally, on a feature set of a few high-cardinality ID fields, the useful interaction
is mostly the `user x item` cross that FM already models directly. Explicit higher-order
crossing has less to find than it does on the wide, many-field ad datasets these papers
were evaluated on.
