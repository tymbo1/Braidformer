# BraidFormer architecture hypothesis

## Scope

BraidFormer-H1 is a hypothesis that a sequence model can obtain a useful quality/compute trade-off by making recurrent or state-space processing the majority path, while using selective mechanisms for exact retrieval and conditional computation. The hypothesis is intentionally open to falsification. A simpler control may be better.

The planned ingredients are:

1. recurrent/state-space sequence processing as the majority backbone;
2. periodic or selective attention for exact retrieval and correction;
3. a sparse latent mixture-of-experts (MoE);
4. adaptive or conditional depth;
5. a unified compute controller that can allocate work across attention, expert count, depth, and deliberation;
6. optional multi-token prediction;
7. later compressed landmark memory;
8. later conditional/static memory;
9. eventual integration with Sovereign Memory.

No item above is considered beneficial until it beats an appropriately matched control or reveals a clearly documented trade-off.

## Initial model scale

The first model family is BraidNano, targeted at approximately 40M–60M parameters. It is a cheap instrument for architecture falsification, not a proxy for frontier-scale performance. Parameter counts, active parameters, and approximate training FLOPs must be measured rather than inferred from labels.

The initial planning configuration uses a 512-dimensional representation, 8 layers, 8 heads for attention-capable controls, a 2,048-dimensional feed-forward block, a 32k vocabulary, and a 512-token context. These values are a conservative starting point for the audited 8-GiB-class GPU, not a validated performance or memory result. The configuration is planning-only until CUDA is functioning and a memory smoke test passes.

## Controlled decomposition

The architecture ladder introduces one major capability at a time:

- A0 establishes the conventional Transformer control.
- A1 changes the majority sequence-processing path to recurrent/state-space processing.
- A2 adds periodic/selective attention for retrieval and correction.
- A3 adds a small sparse latent MoE.
- A4 adds conditional depth.
- A5 adds the unified compute controller.
- A6–A8 investigate memory and multi-token prediction extensions after the core comparison is viable.

Implementation should expose each mechanism behind an explicit configuration field. A configuration must identify the stage and must not silently activate later mechanisms.

## What would falsify the hypothesis?

Examples of falsifying evidence include:

- the recurrent/state-space majority backbone cannot match the A0 control at matched budget;
- selective attention does not recover the long-range retrieval failures it is intended to address;
- MoE or conditional depth adds overhead without a compensating quality or efficiency benefit;
- the controller makes unstable or non-reproducible allocation decisions;
- memory variants improve a narrow probe but lose on held-out tasks or total compute;
- any apparent gain disappears after matching tokens, FLOPs, context, optimizer, and evaluation.

These are evaluation criteria, not predictions.

## Phase 0 implementation boundary

This phase provides repository conventions, diagnostics, configuration validation, and documentation only. It does not implement all architecture stages, download data, download checkpoints, or start training.
