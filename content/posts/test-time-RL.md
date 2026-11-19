---
title: "Test-Time Scaling for RL: A Corollary to Never Give Up"
date: 2026-11-20
description: "I reframe my last paper with a new perspective: test-time search for RL. This allows me to implement it in a new library, Miles."
math: true
toc: true
tags: ["research", "post-training"]
---
{{< epigraph author="My friend Samuel Lavoie, paraphrased" >}}
I think it takes time to understand your own paper. A year after publishing [SEM](https://arxiv.org/abs/2204.00616), I understood what it was really about.
{{< /epigraph >}}

I recently released a paper on RL post-training of LLMs introducing the Matthew effect and proposing to solve it with [Never Give Up](https://mnoukhov.github.io/posts/ngu/). As with every paper, I get a lot of interesting links and feedback after releasing it. After letting it settle, I think I understand my own paper better. This post takes another look at my ideas in the paper and finds three cool new connections:
1. The same Matthew effect, in MoEs
2. A more general formulation of NGU as test-time scaling for RL
3. A practical implementation of test-time scaling in an RL for LLM library, Miles

Overall, this ended up informing how I implemented NGU in the [Miles](https://github.com/radixark/miles/) library [(PR submitted!)](TODOLINK) and even slightly changed the actual method, to better fit the framework. You don't need to have read my other blog post/paper to understand this one, but it might make things more interesting.


## The Matthew Effect in MoEs

Briefly summarizing, I found that RL on LLMs tends to improve performance on the problems the LLM is already good at. This means easier problems get much easier and hard problems get harder. I called this *the Matthew effect* after the network science / economics phenomenon known as "the rich get richer".

After publication, I found the Matthew effect has been previously noticed in subfields similar to RL{{< sidenote >}}[Chen et al (2025)](https://arxiv.org/abs/2502.18001) find that more "capable" models benefit more from distilling high-quality CoTs, but this is confounded because their models are also larger, not just better. [Guo, Xi et al (2026)](https://arxiv.org/abs/2510.26474) find this happens for iterative SFT on VLMs, similar to RL for LLMs.{{< /sidenote >}} but imagine my surprise listening to a lecture on MoEs and hearing

{{< blockquote author="Tatsu Hashimoto" cite="[CS 336, Lecture 4](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3130s)" >}}
[In MoEs] you get this sort of *rich get richer* effect where experts that are chosen get strong weights. Strong weights mean that they're selected more often and they run away.
{{< /blockquote >}}

Training MoEs, in general, seems to suffer from a Matthew effect of its own. In RL for LLMs, easy problems are more likely to get signal and allow for improvement. In MoEs, it is the already-likely experts that are routed to and get reinforced.

It's been observed that this is even more pronounced when training MoEs with RL. MiMo-V2.6-Pro's open RL run{{< sidenote >}}MiMo [live-streamed](https://mimo.xiaomi.com/rl/) their RL training openly showing metrics, cost, and performance. It was both a great demonstration of open science and something akin to [a live sporting event](https://x.com/srush_nlp/status/2100427133440950536?s=20) for the ML community. {{< /sidenote >}} on a 1T / 42B-active model shows that training the MoE router can *load-collapse* experts leading to some experts becoming unused ("cold") while others take on much more of the load.

{{< figure src="/images/ngu_more/mimo-rl-moe-collapse.png" alt="MoE router collapse during MiMo RL" caption="from [MiMo-V2.6](https://arxiv.org/pdf/2610.11959v1)" >}}

The current solution used in MoE RL seems to be just avoiding the problem altogether and freezing routing{{< sidenote >}}This is true for [Nemotron 3.5](https://github.com/NVIDIA-NeMo/RL/blob/super-v3.5-posttraining/examples/nemo_gym/nemotron-3.5-super/student_rlvr.yaml#L642) and the aforementioned MiMo-V2.6-Pro but not clear for GLM or DeepSeek. Also note that routing can change due to hidden weights changing even if router weights remain the same.{{< /sidenote >}} although load-balancing auxiliary losses have been suggested too.

In RL, I focus on reducing compute on easy problems and implicitly transferring leftover compute to more difficult tasks. For MoEs, this suggests training routers and figuring out how to use less compute for frequently chosen experts. This doesn't feel like a super clear approach and it's possible that freezing the router is simply a sufficient solution. Overall, I don't have the answer here. I just think it's interesting to see there are Matthew effects across the LLM stack, and the danger of their compounding effect.

## NGU as Test-Time Scaling for RL

That was a new view of the problem, now onto a view of my solution, NGU, that I think helps shape the future of RL infra. The standard approach in GRPO is to take `$K$` samples per prompt and hope one of them gets the right answer.

{{< figure src="/images/ngu_more/grpo-samples.svg" alt="GRPO sampling: a prompt goes to the model, which samples K completions, each marked correct or incorrect" caption="GRPO takes `$K$` samples per prompt, each graded right or wrong" >}}

My proposed mitigation of the Matthew effect was to reduce the compute allocated to easy problems by taking fewer samples on them. Quite simply, taking a large number of samples `$K$` for every prompt could be wasting inference compute on easy problems and then wasting training compute on noisy responses where `$\frac{K-1}{K}$` are correct. Instead, I proposed *Never Give Up*: take a fraction, e.g. `$\frac{K}{4}$`, and quickly filter the easy prompts. If a problem wasn't solved in the first pass, take `$\frac{K}{4}$` more samples, and so on.

{{< figure src="/images/ngu_more/ngu-resample.svg" alt="Never Give Up: the model takes K/4 samples, all wrong, so it takes K/4 more, one of which is correct" caption="NGU takes a fraction of `$K$` samples at a time, and only samples more if none are correct." >}}

You know what this looks like? **Test-time scaling via search**.

We can see GRPO's regular approach to inference as purely parallel search.

{{< figure src="/images/ngu_more/tts-parallel.svg" alt="Parallel search: a prompt goes to the model, which samples K completions in parallel" caption="Parallel search" >}}

In contrast, *sequential* search leverages the previous result to inform future sampling.

{{< figure src="/images/ngu_more/tts-sequential.svg" alt="Sequential search: the model produces a completion, which is fed back to the model to produce the next one, for T rounds" caption="Sequential search" >}}

But modern methods for test-time search are much richer. Hybrid search methods combine sampling multiple completions in parallel with iterative sequential refinement. For example, one can put previous failures alongside verifier feedback into the context, in order to avoid the same mistakes.

{{< figure src="/images/ngu_more/tts-hybrid.svg" alt="Hybrid search: the model samples K completions in parallel, which are fed back to the model to sample K more" caption="Hybrid search, figure based on [Venkatraman et al (2025)](https://arxiv.org/abs/2509.26626)." >}}

Given that we're trying to solve harder problems, it makes sense to use more advanced inference techniques. But how can we practically implement test-time methods in an already-complex RL pipeline? Next, I integrate NGU as a test-time search into a powerful open-source RL library, [Miles](https://github.com/radixark/miles).

## Implementing Test-Time Scaling in RL

In implementing NGU in the Miles library, I was trying to find the right coding paradigm for both my method and extensibility for future test-time search methods. At the same time, I wanted a module that could fit into Miles' current data flow with as few changes as possible. 

I figured that a fundamental part of NGU and future test-time scaling is how we *leverage previous (failed) completions*. Whereas NGU simply notes previous failures, future methods can explicitly use previous, failed completions to do better in future iterations.{{< sidenote >}}It turns out a version of this idea was concurrent with NGU. A one-off baseline by [Zheng et al, (2026)](https://arxiv.org/abs/2605.28751) ("online-fixing" in Appendix G) puts a previous failed attempt in the prompt alongside instructions to improve on it. It works well!{{< /sidenote >}}

If our RL methods are no longer taking a fixed number of samples `$K$`, but iteratively improving and adapting, we also have to move towards a more flexible paradigm for *inference budget* `$B$`. Recently, SOTA open-source agentic evaluations have started talking not about a specific number of outputs (`pass@k`) but rather a timeout that accounts for agentic interactions.{{< sidenote >}}[MiniMax M3](https://www.minimax.io/blog/minimax-m3) eval has a 25 min timeout for LiveSQLBench whereas PostTrainBench gets 12 hours.{{< /sidenote >}}  

With these two pieces, we can implement many test-time scaling methods as keeping track of some state, leveraging failed completions, and using up to some inference budget's worth of compute.

## NGU, now in Miles!

Altogether, I implemented NGU in Miles by
1. extending the existing `async_unused_samples_handler` to become a generic module for what to do with groups that were aborted, stale, all-wrong, etc.
2. treating `never_give_up` as an `unused_samples_handler` that takes `$K$` more samples for a failed group up to some maximum budget `$B$`

This implementation leads to two important changes to NGU.

1. NGU as reducing budget, not increasing it

If we set a maximum budget `$B$`, NGU can't probabilistically decide to take more than `$B$` samples. 
Instead, my mental model of NGU is now about iteratively generating `$K$` completions to a prompt, up to some maximum budget `$B$`. Essentially, checking every `$K$` compute if we get a correct response, and early-stopping if the prompt was easy enough.
Despite the name, *Never Give Up*, this treats NGU as a method of *reducing* inference below some maximum compute.{{< sidenote >}}Maybe I should rename it *Give Up Early and Often*{{< /sidenote >}} 

2. Inference budget can be completions, *or tokens*

As mentioned, our budget `$B$` doesn't need to be the number of completions but can be GPU hours or even *tokens*. A token budget allows NGU to become a method for reducing *context length* instead of the number of samples. For example, say a response failed to finish an answer within the initial budget of 16k response tokens. Instead of starting a new response, we can also NGU on the number of *tokens* and give the unfinished response more token budget, essentially giving a longer answer.

{{< figure src="/images/ngu_more/ngu-token-extension.svg" alt="NGU with a token budget: y1 runs out of its 16k tokens before answering and the other three samples are wrong, so the model continues y1 from 16k to 32k tokens and starts three new samples y5 to y7; the extended y1 is correct" caption="NGU on tokens: a response that hits the 16k limit (...) is extended for another 16k instead of being replaced" >}}

### Verifying on an RL Benchmark

To double-check this all works, I rerun my previous RL on Math experiments, which still use a sample budget, now in Miles. As NGU is now a method to reduce budget, I compare to GRPO baselines with different inference budgets always using `$N=16$` prompts but varying the number of completions `$K \in \{8,16,32\}$`. In contrast, NGU uses `$N=16$` and `$K=8 \to 32$`, taking 8 samples up to a budget of 32.

{{< plotly data="ngu_more/aime_vs_compute" caption="Qwen 3 4B-base RL on 10k subset of Deepscaler for up to 1000 steps or 100 B300 hours" >}}

You can more or less see that smaller budgets (`$K=8$`) do better early in training but long term, you need larger inference budgets (`$K=32$`) to progress on the hardest tasks. NGU allows improving on the easiest problems early and still pushing on the harder problems later in training. This allows NGU `$K=8 \to 32$` to achieve the same performance as `$K=32$` with about `$40\%$` less compute.

## Conclusion

Altogether, we have now changed from a one-off method into a generic framework enabling test-time scaling methods for RL. Our method, NGU, no longer increases samples but focuses on reducing a generic inference budget. It still works the same way: reduce compute on easy problems, implicitly reallocate that compute to harder problems. This direction has really only come about by getting great feedback from many people and taking the time to reflect and dig deeper on where I think RL is going. Hopefully this enables future research to implement more advanced test-time scaling methods and continue to solve ever harder problems. 

### Acknowledgements

Thanks to [Alex Lin Wang](https://alexwang0317.github.io/) for emailing me with a question about how NGU would do with increasing `$K$`. It directly inspired the budget experiment above.

### Citation

```
@misc{noukhovitch_test_2026,
	title = {Test-Time Scaling for RL: A Corollary to Never Give Up},
	url = {mnoukhov.github.io/posts/test-time-RL},
	author = {Noukhovitch, Michael},
	month = nov,
	year = {2026},
}
```

