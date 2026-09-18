---
title: "Learning to Solve Hard Problems in RL for LLMs by Never Giving Up"
date: 2026-09-15
description: ""
math: true
toc: true
tags: ["research", "post-training"]
---
This is a blog post for my recent paper on RL post-training of LLMs: introducing the Matthew Effect and proposing to solve it with Never Give Up. It is presented interactively and less formally, more like how I give the talk. For a deeper, more technical dive, check out the paper on [arxiv](https://arxiv.org/abs/2609.13443) and code on [github](github.com/mnoukhov/never-give-up).

## What is your eval actually measuring?

Every good RL practitioner has no doubt seen an eval curve go up. Here is the AIME 2025 eval during our RL training of Olmo 3.1 RL-Zero Math{{< sidenote>}}see [Olmo 3.1 blog post](https://allenai.org/blog/olmo3) and [arxiv](https://arxiv.org/abs/2512.13961){{< /sidenote >}} 



{{< plotly data="ngu/olmo3_7b_rlzero_math_accuracy" caption="Training Olmo 3 7B base with RL on Dolci RL-Zero math improves its overall math ability. Or does it?">}}

What does this curve really mean?



Our eval is an average over 30 AIME questions. Let's break those 30 questions down into 3 levels of difficulty.
Every question that our initial, pre-RL model gets 0 for pass@32 will be labelled "hard". The other questions we'll divide evenly by into "medium" and "easy" based on their pass-rates. So our initial pass@1 averages will be 0%, 3.8%, 22.7% for our subsets. How do you think performance on each subset will evolve?

{{< plotly data="ngu/olmo3_7b_rlzero_math_accuracy_by_difficulty" animation-speed="2" play-style="pale-green" caption="Our AIME evals during Olmo 3.1 RL-Zero split into three levels of difficulty by their initial accuracy. Easy AIME problems improve drastically but problems that start with pass@32=0 mostly end with pass@32=0!">}}



Averaging our AIME eval was hiding something important: the majority of our improvements are coming from the easiest problems going from somewhat solved to mostly solved. The hardest problems are barely improving. This is clearly visible if you look at how each example's solve rate changes over time (see plot in the margin). 
{{< marginfigure src="images/ngu/aime_per_example_solve_rate.svg?v=titleless" alt="Heatmap of solve rate for each AIME evaluation example across training steps, grouped by initial difficulty." caption="**Accuracy of each AIME eval example over training.** We order examples by difficulty from top (initial model pass@32=0) to bottom (initial model pass@1 > 30%). The hardest examples (top rows) barely improve over training. The model mainly learns to better solve easy and medium-difficulty examples that were already reasonably-well solved. We call this discrepancy the **Matthew Effect**." >}}
But this is for math RL on LLMs. What about other domains?

We evaluate code RL and agentic RL using [Deepcoder](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51) and [DeepSWE](https://pretty-radio-b75.notion.site/DeepSWE-Training-a-Fully-Open-sourced-State-of-the-Art-Coding-Agent-by-Scaling-RL-22281902c1468193aabbe9a8c59bbe33), two nice open-source projects that released models and logs.
We can use the initial model to split each benchmark into difficulty buckets (Deepseek-R1-Distilled-Qwen-14B on LCBv5) or we can use existing task length/difficulty labels (SWEBench).

{{< plotly data="ngu/matthew_main" toggle="split" caption="">}}

The gains from RL are proportional to how easy the problems are. We connect this bias to a similar phenomenon in network science and economics, the *Matthew Effect*{{< sidenote >}}[Merton (1968)](https://www.science.org/doi/10.1126/science.159.3810.56), also see [Wikipedia](https://en.wikipedia.org/wiki/Matthew_effect){{< /sidenote >}}, generally summarized as "the rich get richer". 

We therefore propose **The Matthew Effect in RL for LLMs**

{{< blockquote author="The Matthew Effect in RL for LLMs" cite="" >}}
RL improves performance on a task in proportion to a model’s initial competence—making easy tasks easier while hard tasks often remain difficult.
{{< /blockquote >}}

## What causes the Matthew Effect?

You might assume the issue has to do with GRPO. If we just don't get a correct answer to our problem in our `$k$` sampled completions, then we don't get any gradient and can't improve on this problem.{{< sidenote>}}[Xiong et al (2025)](http://arxiv.org/abs/2510.04996) call this *signal loss*{{< /sidenote >}} One possible answer is to sample more completions i.e. larger `$k$`.{{< sidenote >}}Other approaches include using [priveleged information](https://arxiv.org/abs/2410.05434) and [curriculum learning](http://arxiv.org/abs/2506.02177). These are generally complimentary to our approach.{{< /sidenote >}}

To test this out, we train Qwen 2.5 0.5B Instruct with GRPO on GSM8k platinum and test on the same. We split our dataset into difficulty levels using initial pass@1: easy (25%), medium (10%), hard (5%), and extra-hard (0%) problems. We vary `$k \in \{4, 8, 16, 32\}$` but keep batch size fixed.

{{< plotly data="ngu/gsm8k_per_difficulty_eval" caption="**Smaller k solves harder problems than bigger k**. Our GSM8k eval, split by initial difficulty, as previously. Surprisingly, `$n=64$` prompts and `$k=4$` completions is a better setting for solving hard problems than `$n=8$` prompts and `$k=32$` completions.">}}

It turns out that smaller `$k=4$` is actually best! Why is this happening? 

Lets look at what our training batch is actually composed of. Since we filter any prompt whose completions are all correct or incorrect, our training batch must always be composed of problems with some completions right, some wrong. We plot what percentage of our batch is our easy subset and our extra hard subset and how this changes over time.

Larger `$k$` increases the chance of finding a rare correct solution to a very hard problem. So, naively, we expect it to have more hard problems in the batch. The issue is that larger `$k$` also increases the chances of finding a rare *incorrect* solution to an *easy* problem.

{{< plotly data="ngu/gsm8k_nonzero_prompt_ratios" caption="**GSM8k Platinum, prompt difficulty in training batch**. Tracking the actual prompts that make it into our training batch, by difficulty. `$k=32$` solves more hard problems at the beginning but there's an inflection point at 200 steps after which `$k=4$` overtakes it. ">}}

Early in training, `$k=32$` finds rare solutions to hard problems. But after the inflection point around step 200, `$k=4$` does better. `$k=4$` filters any problem that is solved in `$4/4$` completions. In contrast, for `$k=32$` to filter the same problem, it must be solved much more: `$32/32$`. `$k=4$` ends up spending much less compute on easy problem, especially when they get a rare incorrect solution. The inflection point is when the benefit of finding rare correct answers to hard questions is outweighted by wasting compute training on rare incorrect answers to easy questions.

Because of our asynchronous RL for LLMs setup{{< sidenote>}}[Async RLHF (Noukhovitch et al, 2025)](https://arxiv.org/abs/2410.18252) is a blatant self-citation but also the first async RL for LLMs paper. See also [PipelineRL (Piche et al, 2025)](https://huggingface.co/blog/ServiceNow/pipelinerl){{< /sidenote >}}, all the compute we save filtering easy problems is used to train on harder problems. We argue the issue behind the Matthew Effect isn't just undersampling for hard problems, but spending too much compute on easy problems.{{< sidenote>}}In contrast to signal loss, we term this *signal efficiency*{{< /sidenote >}}

## Never Give Up on hard problems

Our goal is therefore to only use small `$k$` for easy problems but have large `$k$` for hard problems. We propose a simple, but effective method of adapting asynchronous RL sampling: **Never Give Up**. We start sampling some small amount `$k$`. If a prompt is solved within the first `$k$` completions, train on it! If a prompt is fully solved in `$k/k$` completions, then we can easily and quickly filter it.

The tricky part is if all completions are wrong. With probability `$p$`, we *never give up* and add the prompt back to our generator in order to sample `$k$` more completions. We keep track of our old completions and when we do solve the problem, train on our whole `$k * \text{rounds of NGU}$` completions. This creates a geometric distribution for the number of samples we take: if we never solve the prompt, we expect to take `$\frac{ \ \ k}{1-p}$` samples, in expectation.

<maybe NGU animation>

This method is implicitly adaptive. Whereas curriculum learning pre-sets the difficulty of a problem, we find that online, adaptive methods do better as easy problems can become more difficulty over training and vice-versa. On GSM8k, `$k=4$` with NGU `$p=0.9$` outperforms all values of standard GRPO with varied `$k$`. This is especially evident on the hardest subset.

{{< plotly data="ngu/gsm8k_per_difficulty_eval_with_ngu" caption="**Never Give Up with k=4 outperforms all possible values of k**. On GSM8k platinum, NGU with k=4 solve more hard problems than k=4 with the same compute, without degrading performance on easy problems.">}}

It does this by achieving the best of large `$k$` early in training and small `$k$` late in training.

{{< plotly data="ngu/gsm8k_nonzero_prompt_ratios_with_ngu" caption="**Never Give Up gets the best of both k=4 and k=32**. NGU allows `$k=4$` to keep retrying hard problems, solving them as well as `$k=32$` early in training. NGU doesn't overspend compute on easy problems, filtering them as fast as `$k=4$` late in training.">}}


## Async RL staleness and tricks for NGU

Astute readers might already see a downside to the method: stale completions. This section introduces two tricks for dealing with staleness, but its not necessary to the main message so feel free to skip it.


If we take multiple rounds of NGU to get one correct completion, our initial `$k$` completions will be pretty stale by the time we train on them. Stale negatives are known to be bad for LLMs and RL{{< sidenote >}}Async RLHF argues that data staleness slows down training, this was also true [for deep RL](https://arxiv.org/abs/1912.06680). [Le Roux et al (2025)](https://arxiv.org/abs/2503.14286) show that stale negatives are particularly bad. {{< /sidenote >}} so its important to filter completions to be below some age threshold.

{{< plotly data="ngu/gsm8k_per_difficulty_eval_ngu_max_age" >}}

`$T=4$` wins out but this leaves another issue: our GRPO baseline. Just because we don't train on a stale completion doesn't mean we shouldn't use it in our GRPO baseline. Suppose we have 4 stale negative completions, 3 new negatives and just 1 new positive. We should treat our positive as the rare phenomenon it is and set our GRPO baseline to `$\frac{1}{8}$`. But if we only train on the newest 4 completions, our total group's reward becomes non-zero `$\frac{7}{8} - \frac{1}{8} - \frac{1}{8} - \frac{1}{8} = \frac{3}{8}$`. Our options are to ignore the filtered completions from our baseline (*ignore*), to leave the baseline non-zero (*no rescale*), or to anchor the positive and rescale the negative advantages by `$\frac{7}{3}$` to maintain total reward 0 (*anchor pos*). 

{{< plotly data="ngu/gsm8k_per_difficulty_eval_ngu_baseline" >}}

Overall, it makes sense to use all the samples you have for your GRPO baseline, even if you're not training on them.{{< sidenote >}}This baseline + rescaling may be generally useful for async RL if there's filtering of samples for being too off-policy. {{< /sidenote >}}

## NGU at a bigger scale: Math

We scale up to a bigger math RL setup: [DeepScaler](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) with Qwen 3 4B base.{{< sidenote >}}generally following the setup of [Li et al (2025)](https://arxiv.org/abs/2509.02534){{< /sidenote >}} On top of a strong GRPO `$k=16$` baseline, NGU further improves performance, especially on the hardest subsets of our AIME + BRUMO 2025 eval.


{{< plotly data="ngu/deepscaler_gradnorm1_ngu_pass_at_1_improvement_by_difficulty_combined" caption="**NGU outperforms GRPO when scaling up to a realistic math task, DeepScaler.** Final evals on a combination of AIME 2025 and BRUMO 2025 math datasets after training with Qwen 3 4B base for ~120 H100 hours. As previously, the evals are split by initial model's pass@64 (i.e. difficulty) to show that NGU improves on the hardest problems." >}}


The Matthew Effect still persists, but we can mitigate it; NGU helps solve harder questions without really degrading on easier ones. 

## NGU on a different scale: Code

Code RL is fundamentally different from math because math usually has binary verifier: right or wrong. A coding problem has many *tests* and the tests can vary from easy and difficult within a single coding problem. Math problems are either easy or hard. Solving a single coding problem means solving *both* easy and hard tests.

We look at a particularly tough setup: [Manufactoria](https://www.kongregate.com/en/games/pleasingfungus/manufactoria).{{< sidenote >}}following the benchmark setup by [Sun et al (2025)](http://arxiv.org/abs/2509.21016){{< /sidenote >}} Standard GRPO improves performance but eventually stagnates: improving on some tests but failing to pass *all* tests. Splitting the tests by difficulty, we see a clear Matthew Effect.


{{< plotly data="ngu/manufactoria_matthew_after_adaptation" caption="**The Matthew Effect can explain stagnating code RL in Manufactoria**. We combine all tests from all problems and split them by again by initial model's pass@32 difficulty, as before. RL improves easy tests to nearly fully solved. Some hard tests are solved as well. But mostly, training signal is oscillating medium-difficulty tests.">}}


Easy tests are nearly fully solved and improvement on hard tests stagnates. GRPO dedicates the majority of training signal to repeatedly revisiting partially-solved medium tests and oscillates between solving them slightly more or less.{{< sidenote >}}We can see this as just another issue of *signal efficiency*, but for code RL. {{< /sidenote >}}

If our first `$k$` completions all pass `$\frac{7}{12}$` tests, then Never Give Up won't accept the next `$k$` completions unless they pass *more* than `$\frac{7}{12}$`, pushing the model to iteratively do better. Where standard GRPO stalls, GRPO + NGU keeps solving harder and harder tests until it starts to fully pass all tests for a given problem.

{{< plotly data="ngu/manufactoria_grpo_vs_ngu" caption="**NGU allows RL to fully solve coding problems in Manufactoria.** Overall performance (*combined*) stalls in standard GRPO but not in NGU. That's because NGU continues improving on the hardest subset of tests (*hard*) and then starts pass all the tests for some eval problems (*all-tests*).">}}

## The Matthew Effect is a Primacy Bias, sort of

Some *very* astute readers may have noticed that the Matthew Effect resembles a [primacy bias](https://en.wikipedia.org/wiki/Serial-position_effect) where LLMs are predisposed to solve certain problems according to their initial state. This intuitively connects the Matthew Effect to the Primacy Bias in Deep RL{{< sidenote >}}[Nikishin, Schwarzer, D'Oro et al, (2022)](https://arxiv.org/abs/2205.07802) {{< /sidenote >}} where deep RL training runs could be derailed due to bad early samples. This was due, in part, to issues of plasticity in neural networks trained with RL.

Could it be that the Matthew Effect in RL for LLMs caused by plasticity? In short, no.

We start from a post-GRPO checkpoint at 6000 steps that has been stagnant for at least 3000 of those steps. We can then either train with a single reward for passing all tests (*all-tests*){{< sidenote >}}This was how [Sun et al (2025)](http://arxiv.org/abs/2509.21016) originally got around the issue of stagnation on Manufactoria.{{< /sidenote >}} or keep our per-test reward and just add NGU (*per-test NGU*).

{{< plotly data="ngu/manufactoria_reward_compute_time" caption="**The Matthew Effect is not caused by issues of platicity, using code RL on Manufactoria**. Continuing from a very stagnant checkpoint, we can recover performance either by switching to a reward only if *all-tests* pass, or using NGU with the original partial reward *per-test* passed.">}}

Both methods recover strong performance, even after spending a while on suboptimal data. This demonstrates that plasticity is not a major issue and LLMs can generally recover from early bad samples.

## Limitations

The intuition behind Never Give Up is that our RL training can reallocate compute by quickly filtering easy problems. So if a task leans heavily towards very difficult problems, Never Give Up will likely not be effective. NGU's sample, wait, sample more process can take longer to finish a whole group of completions as compared to sampling `$\frac{k}{1-p}$` from the start. This means your early samples will be more off-policy than necessary, which means a worse learning signal and slower learning speed. But if you already know a decent `$k$` for your data distribution, NGU can likely be effective!

## Conclusion

We have highlighted *the Matthew Effect* in RL for LLMs, and shown that standard RL results in disproportionately poor performance on the hardest problems. 
This demonstrates how simple scalar values may not be sufficient for accurate evaluations of LLMs. Hopefully, it inspires you to dig into your evals and look at more dense evaluation signals. 

Our proposed solution, *Never Give Up*, represents a simple method for better allocating compute. The trick is really just reducing the compute spent on easy problems, which reallocates it towards harder problems. But it does seem to show substantial improvements on difficult tasks. Future work should examine more complex multi-step, agentic environments and try to gain a deeper understanding of how RL actually changes our model's distribution, in practice. 


### Acknowledgements

Thanks so much to my coauthors: Hamish, Nathan, and Aaron! And all my friends who I annoyed for advice during the project: Costa, Sam, Finbarr, Dima, and Adrien. Hamish, Nathan, and Adrien kindly gave me feedback on this blog post. Thanks to Ai2 for giving me compute, some really nice stickers, and great colleagues.

All graphs here made with plotly thanks to my robot friends Sonnet and Sol.

### Citation

```tex
@misc{noukhovitch_ngu_2026,
	title = {Learning to Solve Hard Problems in RL for LLMs by Never Giving Up},
	url = {https://arxiv.org/abs/2609.13443},
	author = {Noukhovitch, Michael and Ivison, Hamish and Lambert, Nathan and Courville, Aaron},
	month = sep,
	year = {2026},
}
```
