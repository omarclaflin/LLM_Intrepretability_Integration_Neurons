# LLM Feature Integration Analysis

This project investigates whether Large Language Models (LLMs) encode *feature integration* in addition to feature identity (SAE features). 

"Do LLMs encode feature integration, in addition to feature identity?" as discussed in [Information Space Contains Computations, Not Just Features](https://omarclaflin.com/2025/06/14/information-space-contains-computations-not-just-features/) (much more detailed discussion of the motivations for this project).

Summary: This is largely a statistical validation/indication that this phenomena (e.g. 'meaningful feature interactions') exist, although still early stage. I will follow-up with a more demonstrative (mechanistic intrepretability) repo shortly, with more optimizations, and a more complete workflow.

## Project Overview

The residual error from SAE reconstruction was mapped with a non-linear method (Neural Factorization Machine, NFM), and showed that the residual reconstruction space contains significant (+3-12% relative reconstruction improvement) *non-linear mapping,* that, importantly contributes to the accuracy of the neuron network activation reconstruction. As a baseline control, another (residual) SAE was used instead but demonstrated no contribution (0.1%).

## Results

Summary: Basically, an NFM contributes 3-12% relative loss reduction (improved accuracy) on a SAE reconstruction, in accordance with my hypothesis that features may have meaningful non-linear interactions in neuron activation space. 
I might add later to this section later, listed here for now: (https://omarclaflin.com/2025/06/14/information-space-contains-computations-not-just-features/)


## Technical Approach
1. Train SAE  (part1_sae_implementation_focused_large_wiki.py)
2. Train a secondary (residual) SAE on the reconstruction error (part2_residual_SAE_implementation.py): Null, control
3. Train a NFM on residual/recon error: part2b_residual_NFM_implementation.py, ~6 attempts w/ diff parameters?
   part2c_residual_NFM_interaction_only_implementation.py -- failed attempt at interaction only
4. part3_inspect_components.py -- look at NFM components to see if interaction component learned anything
5. part4_NFM_sparsity_inspector.py -- find top K features (in NFM), top SAE features
6. part5_find_feature_meaning_large_wiki.py -- investigate top SAE features for semantic identity (via existing dataset, clamping, using Claude API)
7. part6_find_strongest_feature_interacting.py -- find other top SAE features interacting within top/targetted K features
8. (part7) -- rerun part5 script on the new SAE feature being targetted 
9. part8_stimulus_response_analysis.py -- chart/graph generation; inputing a 2x2 design, two targetted SAE, targetted K; outputs linear and non-linear effects of the modulation of both SAE features (via input prompts)
10. part9_stimlus_response_listOfDiscoveredFeatures_analysis.py -- refactored part8 script; Takes in stimului design ONLY, finds top SAE features (top N, which can be set to 1, finds top K features (via variance, weight contributions, or difference), and does the same analysis/chart output as part8


As an example, differential effects between two top-activating feature categories (Paris, #32806)/(Folk, #23360) interactions vs. individual 'main effect' of the features).
A stimulus-driven discovery approach generates artificial prompts to probe specific feature interactions, then discovers the underlying SAE patterns post-hoc.

## Implication Summary
This work suggests current interpretability methods like SAEs may be incomplete, as they capture only linear combinations and miss important interaction effects. The findings may explain why polysemantic neurons are robust and prevalent - they perform integration work rather than just storage.

## Relevance/Related Phenomena:

Some prior observations that may be explained by 'feature integration'/'feature interactions':

- Some insight on 'interference' and 'noise'

- Why polysemantic neurons are so robust (they're doing integration, not just storage)

- Why SAEs miss some variance (they ignore integration)

- Possibly why some less intrepretable features (e.g. perhaps High Frequency Latents contain a large set of non-linearly integrated subfeatures?)

- A more intrepretable space for the observed polysemantic neurons

- One potential source for the 'dark matter' of computation

- Insights/approaches into nuanced feature interactions: feature interactions related to safety, bias, alignment, etc

- Potentially how 'binding' works in neuroscience/artifical neuroscience?


## More to do
listed in the article: "Do LLMs encode feature integration, in addition to feature identity?" as discussed in [Information Space Contains Computations, Not Just Features](https://omarclaflin.com/2025/06/14/information-space-contains-computations-not-just-features/)

