---
layout: default

title: > 
  ExoViP: Step-by-step Verification and Exploration with Exoskeleton Modules for Compositional Visual Reasoning
venue: CoLM 2024
authors:
    - name: Yuxuan Wang
      tag: 1
      url: https://patrick-tssn.github.io/
    - name: Allan Yuille
      url: https://www.cs.jhu.edu/~ayuille/
      tag: 2
    - name: Zhuowan Li
      url: https://lizw14.github.io/
      tag: 2, <i class="fa fa-envelope"></i>
    - name: Zilong Zheng
      url: https://zilongzheng.github.io
      tag: 1, <i class="fa fa-envelope"></i>
affiliations:
    - name: BIGAI
      tag: 1
    - name: Johns Hopkins University
      tag: 2
misc: > 
  <sup><i class="fa fa-envelope"></i></sup> Corresponding authors.

arxiv: https://arxiv.org/abs/2408.02210
code: https://github.com/bigai-nlco/ExoViP
---


<section class="hero teaser">
  <div class="container is-max-desktop">
    <div class="hero-body">
    <figure class="image">
      <img src="{{ '/assets/img/framework_latest.png' | relative_url }}" />
      <figcaption><span class="dnerf">Figure 1.</span> <b>An overview of ExoViP.</b> The prediction after each step is verified by the proposed  "exoskeleton" verification modules, which contain a mix of three sub-verifiers. The verified scores help correct the errors in the vision module predictions or refine the reasoning programs planned by LLM.</figcaption>
    </figure>
    </div>
  </div>
</section>



<section class="section">
    <div class="container is-max-desktop" markdown="1">

## Abstract
{:.title}


Compositional visual reasoning methods, which translate a complex query into a structured composition of feasible visual tasks, have exhibited a strong potential in complicated multi-modal tasks. Empowered by recent advances in large language models (LLMs), this multi-modal challenge has been brought to a new stage by treating LLMs as few-shot/zero-shot planners, i.e., vision-language (VL) programming.
Such methods, despite their numerous merits, suffer from challenges due to LLM planning mistakes or inaccuracy of visual execution modules, lagging behind the non-compositional models.
In this work, we devise a "plug-and-play" method, ExoViP, to correct errors in both the planning and execution stages through introspective verification. We employ verification modules as "exoskeletons" to enhance current VL programming schemes. Specifically, our proposed verification module utilizes a mixture of three sub-verifiers to validate predictions after each reasoning step, subsequently calibrating the visual module predictions and refining the reasoning trace planned by LLMs. 
Experimental results on two representative VL programming methods showcase consistent improvements on five compositional reasoning tasks on standard benchmarks. In light of this, we believe that ExoViP can foster better performance and generalization on open-domain multi-modal challenges.
    
        
</div>
</section>

<section class="section" style="background-color:#efeff081" >
    <div class="container is-max-desktop" markdown="1">


## ExoViP
{:.title.is-3}

### Exploration with Verification
{:.title.is-4}

<div class="columns is-centered has-text-centered">
<div class="column is-four-fifths">
<figure class="image">
  <img src="{{ '/assets/img/exovip_modules.png' | relative_url }}" />
</figure>

</div></div>




<div class="columns  is-centered">
<div class="column content" markdown="1">

- **Image-text matching verifier** 

$$s_{ans}^{itm} = \textrm{ITM}(\mathcal{T}_{ans}, img)$$

- **Image captioning verifier** 

$$s_{ans}^{cap} = \textrm{sim}(\mathcal{C}_{ans}, \mathcal{C}_{img})$$

</div>

<div class="column content" markdown="1">

- **Visual question-answering (VQA) verifier**

$$s_{ans}^{vqa} = \textrm{VQA}(\mathcal{Q}_{ans}, True) - \textrm{VQA}(\mathcal{Q}_{ans}, False)$$

- **Negative Sampling** We take the difference of a candidate answer $$ans$$ with its semantic opposites $$neg$$ as the final verification score, *i.e.*, $$s = s_{ans} - s_{neg}$$.

</div>
</div>


### Exploration with Reasoning Trace
{: .title.is-4}

<div class="columns">
<div class="column content" markdown="1">

**Tree-based reasoning trace searching (TRS)** The reasoning trace searching procedure is represented as a tree structure, where each node is a reasoning operation. To get a better reasoning trace, we search from the tree using the beam search algorithm.

**Post-hoc self-correction (PSC)** We use the self-correctness ability of LLMs to rank the $$K$$ traces and select the top $$P$$ from them ($$P<K$$). 



</div>
<div class="column">
<figure class="image" >
  <img src="{{ '/assets/img/ExoViP_search.png' | relative_url }}"  />
</figure>
</div>
</div>


</div>
</section>

<section class="section">
    <div class="container is-max-desktop" markdown="1">

## Downstream Tasks
{:.title.is-3}

<div class="columns is-centered has-text-centered">
<div class="column">

<figure class="image" >
  <img src="{{ '/assets/img/exovip_case_gqa.png' | relative_url }}"  />
</figure>
<figurecaption><span class="dnerf">Task 1.</span> Compositional Visual Question Answering.</figurecaption>

</div>
<div class="column">

<figure class="image" >
  <img src="{{ '/assets/img/exovip_case_refcoco.png' | relative_url }}"  />
</figure>
<figurecaption><span class="dnerf">Task 2.</span> Visual Expression Referring.</figurecaption>

</div>
</div>

<div class="columns is-centered has-text-centered">
<div class="column">

<figure class="image" > 
  <img src="{{ '/assets/img/exovip_case_nlvr.png' | relative_url }}"/>
</figure>
<figurecaption><span class="dnerf">Task 3.</span> Natural Language Visual Reasoning.</figurecaption>

</div>
<div class="column">

<figure class="image" >
  <img src="{{ '/assets/img/exovip_case_kilogram.png' | relative_url }}"  />
</figure>
<figurecaption><span class="dnerf">Task 4.</span> Abstract Visual Reasoning.</figurecaption>

</div>
</div>

<div class="columns is-centered has-text-centered">
<div class="column is-four-fifths">

<figure class="image" > 
  <img src="{{ '/assets/img/exovip_case_agqa.png' | relative_url }}"/>
</figure>
<figurecaption><span class="dnerf">Task 5.</span> Spatial Temporal Video Reasoning.</figurecaption>

</div>
</div>

<div class="columns is-centered has-text-centered">
<div class="column is-four-fifths">

<figure class="image" >
  <img src="{{ '/assets/img/exovip_case_imageediting.png' | relative_url }}"  />
</figure>
<figurecaption><span class="dnerf">Task 6.</span> Image Editing.</figurecaption>

</div>
</div>

</div>
</section>


<section class="section">
    <div class="container is-max-desktop" markdown="1">
    
## BibTex
{:.title}

```bibtex
@inproceedings{wang2024exovip,
    title={ExoViP: Step-by-step Verification and Exploration with Exoskeleton Modules for Compositional Visual Reasoning},
    author={Wang, Yuxuan and Yuille, Alan and Li, Zhuowan and Zheng, Zilong},
    booktitle={The first Conference on Language Modeling (CoLM)},
    year={2024}
}
```

</div>
</section>
