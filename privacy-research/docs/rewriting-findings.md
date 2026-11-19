# Choosing a rewriting model: what we found

*October 26, 2026. Results from nine candidate models on 40 real privacy-policy passages. Speed results added November 18, 2026, and the judging of the llama.cpp rewrites on November 19.*

The app's Rewrite feature turns a privacy-policy passage into plain language. It must run on the user's own computer, and it must not change what the policy promises. This page reports how well nine small AI models do that job, and which ones fit the computers we are targeting.

## In short

- **Qwen3.5-4B is the best fit for our main target, computers with 8–12 GB of memory.** Compressed to 4-bit (explained below), it needs about 3.6 GB. It kept 82% of the passages' key facts in full, and only 2 of its 40 rewrites could mislead a reader.
- **Larger models do a little better but need more memory than our target computers have.** Gemma 4 12B kept 92% of key facts and never misled, but it needs about 8.6 GB. Qwen3-14B kept 78% and misled twice, at about 11 GB.
- **The smallest models are not reliable.** The three models of about 1 billion parameters could mislead in 25–52% of their rewrites. The requirements only ask for labeling, not rewriting, on 2–4 GB computers, so this rules out rewriting on that tier.
- **Most mistakes are small but matter for privacy.** The most common problem is dropping a limiting word: "we do not *knowingly* collect children's data" becomes "we do not collect children's data", which turns a limited promise into an absolute one.
- **The software that runs the model decides its speed, and changes its rewrites (added November 18 and 19).** With the software behind the results on this page, Qwen3.5-4B needed about 12 minutes per passage on a MacBook's processor. With llama.cpp the same MacBook needed about 5 seconds, or under 3 on its graphics chip. The llama.cpp rewrites are shorter and easier to read, but they keep fewer key facts in full: about 70%, against 78% when the same judge read both side by side. (The judge was stricter this time; in October it gave the same rewrites the 82% above.) About as many could mislead a reader. See [How fast does it run?](#how-fast-does-it-run)
- **Two things are still unchecked:** a person on the team has not yet reviewed the list of key facts, and no model has been timed on an actual 8–12 GB laptop.

## What we tested

**The task.** Each model got the same instructions (the `summary-v3` prompt): summarize the passage for an ordinary reader in your own words, keep every condition, exception, number, and time limit, keep words such as *may*, *knowingly*, and *reasonable*, make no promise stronger or weaker, and add nothing the passage does not say.

**The passages.** 40 sections from real app privacy policies, each 86–340 words long and each from a different policy. We used the first 20 while improving the instructions. The other 20 come from policies we never looked at while tuning, so they show whether the results hold on new text. The two sets agreed closely: each model's results were similar on both, and the models ranked in the same order on each.

**The models.** Nine freely available models, grouped by the computer memory they need:

| Size | Models | Version tested |
| --- | --- | --- |
| Small (about 1 billion parameters) | Qwen3.5-0.8B, Gemma 3 1B, LFM2.5-1.2B | Full size |
| Medium (3–8 billion) | Granite 4.2 3B, Qwen3.5-4B, Gemma 4 E4B | Compressed to 4-bit |
| Large (12–14 billion) | Gemma 4 12B, Ministral 3 14B, Qwen3-14B | Compressed to 4-bit |

**Why compressed.** A model is a large table of numbers. Storing each number in 4 bits instead of 16 makes the model about a quarter of the size, at a small cost in accuracy. The app must keep its memory use under half of the computer's total memory: under 4 GB on an 8 GB laptop, or 6 GB on a 12 GB one. At full size, no medium or large model fits under that limit. So we tested them in the compressed form the app would actually use. Compression changed our automatic checks only slightly, with one exception: compressed Qwen3.5-4B sometimes copies the passage instead of rewriting it (see [Does compressing a model hurt it?](#does-compressing-a-model-hurt-it)).

## How we scored the rewrites

We scored every rewrite in two ways. Automatic word checks run instantly on any rewrite and need no judge. An AI judge then read every rewrite for meaning, because word checks cannot tell whether a promise survived.

### Automatic word checks

| Check | What it measures | Why it matters |
| --- | --- | --- |
| Valid | The answer finished, is not empty, and has no leftover "thinking" text | An unusable answer cannot be shown to anyone. |
| Numbers kept | How many of the passage's numbers, such as ages, days, or years, are still in the rewrite | A lost "13" or "30 days" changes what a reader can expect. |
| Limiting words kept | How many of the passage's limiting words are still there: *may*, *might*, *could*, *likely*, *if*, *only*, *unless*, *except*, *not*, *never*, *without*, and *knowingly* | These words limit a promise. Dropping *knowingly* or *not* turns a careful promise into a different one. |
| Limiting words added | How many rewrites add *only*, *unless*, *except*, *not*, *never*, or *without* where the passage has none | An added exception or restriction makes a promise stronger or weaker than the policy does. |
| Reading level | The US school grade a reader needs, from sentence and word length (Flesch–Kincaid) | The point of the feature is easier text. The original passages score about grade 16. |
| Copying | The average length of the word-for-word stretches a rewrite takes from the passage; 1–2 means the model used its own words | A rewrite that copies whole sentences keeps the facts but makes nothing easier to read. |
| Format breaks | An opening such as "Here's the summary:", or formatting such as bold text or lists | The app shows the rewrite as it comes, beside the policy. |
| Length | The rewrite's words, against the passage's | Not ranked: it shows how much a model shortens the text. |

**How the checks shaped the instructions.** The `summary-v3` prompt is our third version, and these checks drove each change:

- With a first, generic prompt, Qwen3.5-0.8B mostly copied the passages back, with a copying score of 97 against 1–22 for the other models. Ministral 3 14B opened 12 of 20 rewrites with "Here's your summary:" and used formatting in 18. The second prompt asked for the model's own words, plain text, and no introduction. Qwen3.5-0.8B's copying fell to 6, and Ministral's introductions stopped.
- Under the second prompt, rewrites added *only* and *unless* where the passage had neither, and dropped *knowingly*. We added the "limiting words added" check and put *knowingly* on the list. The third prompt names these words and forbids making any promise stronger or weaker.

**Why the checks are not enough.** They count words, not meaning, and they count over a whole passage, not fact by fact:

- Four models told readers they would get "30 days' notice" that the policy only promises to a library. The "30" survived, so the numbers check passed.
- Ministral 3 14B said the company uses "encryption, secure storage", protections the passage never names. No limiting word changed, so no check noticed.
- Qwen3-14B kept fewer limiting words than Granite 4.2 3B (72% against 82%), yet the judge found it far more faithful: 78% of key facts kept against 65%, and 2 rewrites that could mislead against 8. Word counts alone would rank these two the wrong way round.

The checks still earn their place. They are the only measure of whether a rewrite is easier to read, they catch copying and format problems the judge ignores, and they are cheap enough to run inside the app, for example to warn when *may* or *knowingly* disappears.

### Key facts and an AI judge

**A checklist of key facts.** For each passage we wrote down the facts a reader needs, 231 in all, such as "The company does not knowingly collect personal information from children under 13." 125 of them are marked **critical**: getting one of these wrong would mislead a reader about their data, their rights, or their risks.

**An AI judge.** Claude Opus 5.5 read each passage, its checklist, and all nine models' rewrites of it side by side. The model names were hidden and the order was shuffled for every passage, so the judge could not tell which model wrote which rewrite. For every fact in every rewrite, it chose one of four verdicts:

- **Kept**: the rewrite says it, with its conditions and limiting words.
- **Partly kept**: the rewrite says it but loses or blurs a part that matters, such as a condition, an exception, or a word like *may* or *only*.
- **Left out**: the rewrite does not say it.
- **Contradicted**: the rewrite says something the passage rules out, such as turning "may" into "will" or adding an exception the passage does not make.

The judge also listed anything a rewrite claims that the passage does not support, and rated it **major** if it could mislead a reader. It had to quote the rewrite's exact words for every problem it reported; a checker rejected any quote it could not find.

**"Could mislead".** We count a rewrite as one that could mislead when it contradicts a critical fact or adds a major claim the passage does not make.

## Results

Both passage sets combined, 40 passages per model:

| Model | Memory needed | Facts kept | Partly kept | Left out | Contradicted | Rewrites that could mislead | Reading level |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B | 1.6 GB | 57% | 32% | 6% | 5% | 10 of 40 (25%) | grade 12 |
| Gemma 3 1B | 2.1 GB | 26% | 51% | 17% | 6% | 21 of 40 (52%) | grade 9 |
| LFM2.5-1.2B | 2.5 GB | 19% | 54% | 19% | 8% | 19 of 40 (48%) | grade 8 |
| Granite 4.2 3B | 3.1 GB | 65% | 26% | 5% | 4% | 8 of 40 (20%) | grade 12 |
| **Qwen3.5-4B** | **3.6 GB** | **82%** | 15% | 3% | 1% | **2 of 40 (5%)** | grade 13 |
| Gemma 4 E4B | 9.7 GB | 87% | 10% | 1% | 1% | 4 of 40 (10%) | grade 11 |
| Gemma 4 12B | 8.6 GB | 92% | 7% | 1% | 0% | 0 of 36 (0%) | grade 13 |
| Ministral 3 14B | 10.1 GB | 68% | 25% | 3% | 4% | 8 of 40 (20%) | grade 8 |
| Qwen3-14B | 11.0 GB | 78% | 19% | 2% | 2% | 2 of 40 (5%) | grade 10 |

- **Memory needed** is the most the model used while rewriting, measured on a graphics card. The whole app will need somewhat more.
- **Reading level** is the US school grade a reader needs, estimated from sentence and word length. The original passages score about grade 16, which is college level, so every model simplified them.
- **Gemma 4 12B** was judged on 36 rewrites, not 40: four of its answers began with a stray, empty "thinking" marker before the summary. The app could strip that marker, but our checks count those answers as invalid.

The charts show the same results for each passage set separately. Each bar splits all of a model's key facts into kept (green), partly kept (yellow), left out (grey), and contradicted (red). The middle panel shows only the critical facts.

![Judged key facts, first 20 passages](../results/comparisons/rewrite-keypoints-v3/keypoints.png)

![Judged key facts, 20 held-out passages](../results/comparisons/rewrite-keypoints-v3-holdout/keypoints.png)

### What the word checks found

Both passage sets combined. The 40 passages hold 20 numbers between them; limiting words are counted over every passage.

| Model | Numbers kept | Limiting words kept | Rewrites adding a limiting word | Copying | Format breaks |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B | 10 of 20 | 81% | 1 of 40 | 6.2 | 0 of 40 |
| Gemma 3 1B | 8 of 20 | 48% | 6 of 40 | 1.8 | 0 of 40 |
| LFM2.5-1.2B | 7 of 20 | 46% | 8 of 40 | 0.9 | 0 of 40 |
| Granite 4.2 3B | 16 of 20 | 82% | 6 of 40 | 3.2 | 2 of 40 |
| **Qwen3.5-4B** | **17 of 20** | 83% | **0 of 40** | 9.1 | 0 of 40 |
| Gemma 4 E4B | 13 of 20 | 86% | 2 of 40 | 3.1 | 0 of 40 |
| Gemma 4 12B | 12 of 20 | 83% | 2 of 36 | 4.7 | 0 of 36 |
| Ministral 3 14B | 8 of 20 | 78% | 7 of 40 | 1.3 | 9 of 40 |
| Qwen3-14B | 11 of 20 | 72% | 5 of 40 | 2.2 | 0 of 40 |

- **The checks agree with the judge about the small models.** Gemma 3 1B and LFM2.5 kept under half of the limiting words and added one to 6–8 rewrites. Qwen3.5-0.8B kept more, partly because it copies more.
- **Qwen3.5-4B never added a limiting word and kept the most numbers**, but it also copies the most. Its rewrites are the hardest to read of the medium models (grade 13).
- **Ministral 3 14B still uses formatting** such as bold text or lists in 9 of 40 rewrites, although the instructions ask for plain text. No model opened with an introduction.

### Every measure, one plot per passage set

These plots put every measure we took side by side, one panel per measure and one bar per model, in the legend's colours. The panels fall into three groups:

- **Automatic word checks** (the first eight panels), described in [Automatic word checks](#automatic-word-checks). The plots call limiting words *qualifiers*.
- **The judge's verdicts** (four panels): key facts kept, critical facts kept, facts contradicted, and rewrites that could mislead. These are the numbers in the table above, shown for each passage set.
- **Running costs** (the last three panels): seconds per passage, peak memory, and passage length in tokens. Time and memory were measured on a desktop graphics card. They compare the models with each other, but they do not predict speed on a laptop. Memory here is in GiB (1 GiB is about 1.07 GB).

The held-out passages contain only 3 numbers between them, so on that plot "numbers kept" moves in steps of a third.

![Every measure, first 20 passages](../results/comparisons/rewrite-keypoints-v3/matrix-overview.png)

![Every measure, 20 held-out passages](../results/comparisons/rewrite-keypoints-v3-holdout/matrix-overview.png)

## What goes wrong, with examples

**Limiting words disappear.** This was the most common problem, and it shows up as "partly kept" or, on critical facts, "contradicted".

> Passage: "We do not **knowingly** collect or solicit personally identifiable information from children under 13."
> Qwen3-14B: "we don't collect personal information from them"

The passage only promises not to collect such data on purpose; the rewrite promises it never happens. Six of the nine models made this same change on this passage.

> Passage (an e-book service run through public libraries): "If we change the policy in a **material** manner … we will provide at least 30 days' notice **to the Library**."
> Gemma 3 1B: "If we change how we collect or use your information, we'll give you 30 days' notice."

The rewrite promises the reader a notice that the policy only promises to the library, and for every change rather than only major ones. Four models told readers they would get this notice.

**Invented promises.** Some rewrites add reassurances the policy never makes:

> Qwen3.5-0.8B: "We do not sell your personal information to third parties." *(The passage makes no such promise.)*
> Ministral 3 14B: "encryption, secure storage" *(The passage names no specific protections.)*
> Granite 4.2 3B: "The data is kept only as needed for these purposes." *(The passage says nothing about how long data is kept.)*

**Reversed meaning.** The small models sometimes flip a statement:

> Gemma 3 1B: "We may disclose personal information to these parties, **with or without your consent**." *(In the passage, your consent or direction is what allows sharing beyond the groups it lists.)*
> Qwen3.5-0.8B: "We protect your privacy by collecting data only when it is **not** strictly necessary." *(The passage says it asks for your consent before collecting data that is not strictly necessary.)*

## By computer size

| Computer memory | App's limit (half) | Recommendation |
| --- | --- | --- |
| 2–4 GB | 1–2 GB | No rewriting. Only Qwen3.5-0.8B fits, and only on a 4 GB computer; a quarter of its rewrites could mislead. The requirements only ask for labeling on these computers. |
| 8–12 GB (main target) | 4–6 GB | **Qwen3.5-4B, compressed.** It fits easily on a 12 GB computer. On an 8 GB one it is tight, and possibly over the limit, once the rest of the app is counted. Granite 4.2 3B also fits, but a fifth of its rewrites could mislead. |
| 16 GB | 8 GB | Nothing larger fits under the limit. Gemma 4 12B needs about 8.6 GB, just over it. Qwen3.5-4B is the safe choice. |
| 24 GB or more | 12 GB or more | **Gemma 4 12B** scored best overall. Qwen3-14B kept fewer facts in full (78% against 92%) but writes more simply (grade 10 against 13). |

Gemma 4 E4B scored well (87% kept) but does not fit the medium tier. Part of the model, a large word table, cannot be compressed by the tools we use, so it needs 9.7 GB even when compressed.

## Does compressing a model hurt it?

Compressing to 4-bit cut memory sharply and changed the automatic word checks only a little, except that Qwen3.5-4B started copying more. The plot below covers the six medium and large models on the 20 held-out passages. Each group is one model, with a bar for each version we ran: full size (BF16, purple), 8-bit (INT8, blue) and 4-bit (NF4, green). Gemma 4 E4B ran in all three.

![Compressed against full size, 20 held-out passages](../results/comparisons/rewrite-precision-v3-holdout/matrix-overview.png)

- **Memory.** 4-bit needs about 60% less memory than full size for Qwen3.5-4B (8.1 to 3.3 GiB) and Granite (7.1 to 2.9 GiB). For the large models it needs about a third less than 8-bit.
- **Word checks.** Most moved only slightly, in both directions. No model's count of rewrites that added a limiting word changed by more than two out of 20. Qwen3-14B added a limiting word to 3 rewrites at 4-bit, against 1 at 8-bit; Granite, Ministral and Gemma 4 E4B added fewer.
- **Copying.** Qwen3.5-4B's word-for-word stretches grew from 4.0 to 9.7 words on average, because at 4-bit it copied some passages almost whole.
- **Speed.** On our graphics card, 4-bit ran about three times faster than 8-bit. This says little about speed on a laptop.
- **Not judged.** The judge graded only the 4-bit versions of these models, so this plot has no key-fact panels. Whether compression loses facts that the word checks miss is untested.

The labels are the models' full download names, so Ministral's ends in "BF16" even above its 4-bit bar. The first 20 passages show the same pattern ([rewrite-precision-v3](../comparisons/rewrite-precision-v3.toml)). There, "numbers kept" swings more, because those passages hold only 17 numbers between them.

## How fast does it run?

*Added November 18, 2026. The MacBook's full runs and the judge's results were added on November 19.*

Every result above came from one way of running a model: the transformers library, with the bitsandbytes library for 4-bit compression, on a desktop graphics card. We have now timed the recommended model, Qwen3.5-4B at 4-bit, on a desktop processor and on a MacBook, and with a second program, llama.cpp. llama.cpp reads its own 4-bit version of the same model (a "Q4_K_M" file).

Average seconds for one passage, fastest first:

| Computer | Runs on | Software | Same three passages | All 20 passages | Tokens per second |
| --- | --- | --- | --- | --- | --- |
| Desktop PC | Graphics card | llama.cpp | 0.4 | 0.6 | 183 |
| MacBook | Graphics chip | llama.cpp | 1.8 | 2.6 | 43 |
| MacBook | Processor | llama.cpp | 3.6 | 5.1 | 22 |
| Desktop PC | Graphics card | bitsandbytes | 4.3 | 7.4 | 19 |
| Desktop PC | Processor | llama.cpp | 7.2 | 10.5 | 11 |
| Desktop PC | Processor | bitsandbytes | 46 | 79 | 1.8 |
| Desktop PC | Processor | bitsandbytes, float32 | 106 | 176 | 0.8 |
| MacBook | Processor | bitsandbytes, float32 | 699 | not run | 0.1 |

- **The computers.** The desktop has an RTX 4090 graphics card and a Ryzen 9 7950X3D processor. The MacBook has an M4 Pro chip and 24 GB of memory.
- **The three passages** are the first three of the first set, and its shortest. They are the only ones the MacBook ran with bitsandbytes.
- **All 20 passages** are the first set. On the 20 held-out passages the first five rows took about the same: 0.7, 2.8, 5.1, 7.1, and 10.9 seconds.
- **Tokens per second.** A token is a word or a piece of one. The rate counts the rewrite's tokens over the whole time for the three passages, which includes reading the passage.
- **float32.** On a processor, bitsandbytes calculates in one of two number formats. The faster one needs a processor feature that the desktop has and many laptops lack. float32 works everywhere, and it is the only one the MacBook ran.
- **Which rows are saved runs.** All but the last: the MacBook's bitsandbytes run is not saved in this repository. The desktop's llama.cpp run on the graphics card used a build of llama.cpp that the project's setup does not install yet.

What we found:

- **llama.cpp was faster in every case we could compare:** about 12 times on the desktop's graphics card, 8 to 17 times on its processor, and about 200 times on the MacBook's processor.
- **With bitsandbytes, a laptop processor is too slow to use.** A passage took 10 to 13 minutes on the MacBook. On a processor, bitsandbytes unpacks the whole compressed model again for every token it writes. We did not try bitsandbytes on the MacBook's graphics chip: our setup for it runs on NVIDIA graphics cards and on processors only.
- **With llama.cpp, the MacBook rewrote a passage in under 3 seconds on its graphics chip and in about 5 on its processor alone.** Both beat the desktop's graphics card running bitsandbytes.
- **Memory.** On the MacBook, llama.cpp peaked at 5.7 GiB on the processor and 3.4 GiB on the graphics chip over 20 passages, against 6.9 GiB for bitsandbytes on three (1 GiB is about 1.07 GB). The requirements allow the app 6 GiB on an 8–12 GB computer, so the processor run fits with little room to spare, before the rest of the app is counted. macOS's own measure of an app's memory, which leaves out file contents it can drop and read again, was lower: 3.5, 0.9, and 5.4 GiB. On the desktop's processor, llama.cpp peaked at 4.2 GiB in one trial, and bitsandbytes at 4.5 GiB, or 5.6 GiB in float32, on one passage each.
- **The MacBook is not one of our target laptops.** It has at least twice the memory and a fast processor. An 8–12 GB laptop still needs a timing.

### What the judge found in the llama.cpp rewrites

*Added November 19, 2026.*

A different 4-bit version of the model writes different text, so the scores in [Results](#results) do not carry over to llama.cpp. The same judge, Claude Opus 5.5, has now graded six sets of Qwen3.5-4B's rewrites of all 40 passages:

- the bitsandbytes rewrites it judged in October;
- the same model run through bitsandbytes again, on the library version the project uses now;
- llama.cpp's rewrites from four runs: the desktop's processor and graphics card, and the MacBook's processor and graphics chip.

For each passage the judge read all six rewrites side by side, with their sources hidden and their order shuffled, under the same instructions as in October.

Both passage sets combined, 231 key facts and 40 rewrites in each row:

| Rewrites of Qwen3.5-4B | Facts kept | Partly kept | Left out | Contradicted | Rewrites that could mislead |
| --- | --- | --- | --- | --- | --- |
| bitsandbytes, the October rewrites | 78% | 18% | 3% | 1% | 3 of 40 |
| bitsandbytes, current library version | 78% | 18% | 3% | 1% | 3 of 40 |
| llama.cpp, desktop processor | 68% | 28% | 3% | 1% | 5 of 40 |
| llama.cpp, desktop graphics card | 68% | 27% | 3% | 1% | 4 of 40 |
| llama.cpp, MacBook processor | 71% | 24% | 4% | 0% | 2 of 40 |
| llama.cpp, MacBook graphics chip | 70% | 26% | 3% | 1% | 4 of 40 |

- **The llama.cpp rewrites keep fewer facts in full: 68–71%, against 78%.** Almost all of the difference moved to "partly kept": the fact is there, but a detail or a condition is gone. The gap is steady. Against each of the four llama.cpp runs, the bitsandbytes rewrite kept more facts in 16 to 19 of the 40 passages and fewer in 6 or 7.
- **About as many rewrites could mislead:** 2 to 5 of 40 with llama.cpp, against 3 of 40. Facts left out or contradicted are as rare in one as in the other.
- **llama.cpp's version shortens more.** Its rewrites average about 100 words against 114, read at grade 12 to 13 against 13 to 14, and copy much less of the passage (a copying score of 3 to 5 against 6 to 11). The bitsandbytes rewrites stay close to the passage's own sentences, which keeps details but simplifies less (see "Copying is not simplifying" under [Limits](#limits-of-this-evaluation)).
- **We cannot yet say which version is at fault.** In the word checks below, llama.cpp's version behaves like the full-size model, and the full-size model has not been judged. So llama.cpp's file may lose facts, or the bitsandbytes version may keep them only because it copies.
- **The judge was stricter than in October.** It gave the October rewrites 78% now and 82% then, and found 3 that could mislead, against 2. It gave the same verdict on 217 of the 231 facts; 11 moved from kept to partly kept. So compare the rows of this table with each other, not with the table in [Results](#results).
- **The current library version did not change the score.** Run through bitsandbytes on torch 2.15.0, the model wrote a different rewrite for 20 of the 40 passages and scored the same.
- **The computer changes the wording, not the picture.** Any two of the four llama.cpp runs wrote exactly the same rewrite for only 3 to 7 of the 40 passages, yet their scores lie within 4 points of each other. That spread shows how far a score moves by chance alone.
- **The judge was consistent within a passage.** Where two of a passage's six rewrites were word for word the same (52 pairs), it gave both the same verdict on every fact.

What the llama.cpp rewrites lose, in examples:

> Passage: "If we make changes, we will **notify you** by revising the date at the top of the policy"
> bitsandbytes: "If changes are made, they will be notified by revising the date at the top of the policy"
> llama.cpp: "The company may update this policy from time to time by changing the date at the top"

The promise to notify is gone. The judge marked this fact partly kept in all four llama.cpp runs.

> Passage: "We will also retain Personal Data and Usage Information for internal analysis purposes and legitimate business interest … or when we are legally obligated to retain this data for longer time periods."
> llama.cpp: "We also retain data for internal analysis and business purposes when legally required for longer periods."

A separate reason became a condition: the rewrite ties keeping data for analysis to a legal requirement. All four llama.cpp runs wrote this sentence, and the judge counted each as a rewrite that could mislead.

> Passage: "(see Your Choices below for information about how to opt out of these communications at any time)"
> llama.cpp on the desktop's processor: "We may opt you out of these communications at any time."

The rewrite reverses who decides. Only this one run made the mistake. The MacBook's processor run kept the opt-out; the other runs, bitsandbytes included, left it out, which the judge marked partly kept.

The charts show the desktop's three runs for each passage set. The MacBook's are in [results/comparisons/rewrite-keypoints-llama-cpp-v3-macbook/](../results/comparisons/rewrite-keypoints-llama-cpp-v3-macbook/) and [its held-out twin](../results/comparisons/rewrite-keypoints-llama-cpp-v3-holdout-macbook/).

![Judged key facts on the desktop, first 20 passages](../results/comparisons/rewrite-keypoints-llama-cpp-v3-desktop/keypoints.png)

![Judged key facts on the desktop, 20 held-out passages](../results/comparisons/rewrite-keypoints-llama-cpp-v3-holdout-desktop/keypoints.png)

**The word checks point the same way.** On the first 20 passages:

| Version of Qwen3.5-4B | Words per rewrite | Copying | Numbers kept | Limiting words kept | Rewrites adding a limiting word |
| --- | --- | --- | --- | --- | --- |
| 4-bit, bitsandbytes (the October rewrites) | 114 | 8.4 | 15 of 17 | 82% | 0 of 20 |
| Full size | 102 | 3.4 | 9 of 17 | 84% | 0 of 20 |
| 4-bit, llama.cpp (desktop processor) | 97 | 3.2 | 9 of 17 | 77% | 1 of 20 |

In these checks the llama.cpp version behaves like the full-size model, not like the bitsandbytes version. It copies much less and writes shorter, easier text (grade 12 against 13). It also keeps fewer numbers: 8 to 11 of 17 across the four runs. Across both passage sets, each llama.cpp run added a limiting word to 1 to 3 of its 40 rewrites, where the bitsandbytes version added none. Every llama.cpp rewrite was valid.

## Limits of this evaluation

- **The checklists have not been reviewed by a person.** Claude drafted them from the passages alone. If the team changes a fact or its critical mark, the affected passages need judging again; the tools refuse to reuse a judgment whose checklist changed.
- **One judge, one reading.** Each passage was judged once, by an AI model. Six separate judging sessions shared the 40 passages, and they handled some borderline wording differently. For example, one counted "should" becoming "must" as kept, while another counted "you must use a strong password" as a contradiction. Within a passage, one judge graded all nine rewrites side by side, so the comparison *between* models is more trustworthy than the exact percentages. A second judging in November shows the size of this: the judge gave Qwen3.5-4B's same rewrites 78% of key facts kept, against 82% in October (see [What the judge found in the llama.cpp rewrites](#what-the-judge-found-in-the-llamacpp-rewrites)).
- **The same AI wrote the checklists and judged.** Both came from Claude, so they may share blind spots. None of the nine candidates is a Claude model.
- **Copying is not simplifying.** A rewrite that copies the passage keeps every fact but helps no one. Compressed Qwen3.5-4B copied two passages almost word for word and leaned heavily on the original wording in a few others. Its rewrites read at grade 13, the hardest of the medium models. This helped its score and is worth watching.
- **Short passages only.** The passages are 86–340 words. Users can paste any length.
- **Speed on a target laptop is unknown.** The judged tests ran on a desktop graphics card. The one laptop timed since, a 24 GB MacBook, is larger and faster than the 8–12 GB laptops the app targets (see [How fast does it run?](#how-fast-does-it-run)).
- **The rewrites came from an older library version.** They were generated with torch 2.8.0. The project now uses torch 2.15.0, which makes the same model write different text for some passages. In a hand check, compressed Qwen3.5-4B wrote exactly its judged rewrite for 10 of the first 20 passages; the rest change partway through, from a few words to most of the rewrite. The scores in [Results](#results) describe the torch 2.8.0 rewrites. Qwen3.5-4B's new rewrites have since been judged and scored the same; the other eight models' have not.

## Next steps

1. **Team review of the key facts**, especially which ones are critical. Then re-judge the passages whose facts changed.
2. **Run Qwen3.5-4B on an 8–12 GB laptop without a graphics card** to measure its speed and the whole app's memory.
3. **Try longer passages**, and the Simplify for Kids and Simplify for Adults styles, with the leading models.
4. **Look for a fix for dropped limiting words**, the most common error, such as an automatic check that warns when *may*, *knowingly*, or *only* disappears.
5. **Have the judge grade the full-size Qwen3.5-4B and a larger llama.cpp file** (5, 6, or 8 bits), to learn whether llama.cpp's 4-bit file loses facts or the bitsandbytes rewrites keep them by copying.

## Where the numbers come from

- Judged results: [comparisons/rewrite-keypoints-v3.toml](../comparisons/rewrite-keypoints-v3.toml) and [comparisons/rewrite-keypoints-v3-holdout.toml](../comparisons/rewrite-keypoints-v3-holdout.toml). The tables combine the two `summary.csv` files that `pbench compare` writes for them under `results/comparisons/`.
- The judge's prompts, replies, and settings: `results/judgments/rewrite/rewrite-keypoints-v3/` and `…-holdout/`.
- Key-fact checklists: `data/rewrite/keypoints/`. Judge instructions: `prompts/rewrite/judge/keypoints.toml`.
- Compression against full size: [comparisons/rewrite-precision-v3.toml](../comparisons/rewrite-precision-v3.toml) and [comparisons/rewrite-precision-v3-holdout.toml](../comparisons/rewrite-precision-v3-holdout.toml).
- Speed on the desktop: [comparisons/rewrite-llama-cpp-v3-torch215.toml](../comparisons/rewrite-llama-cpp-v3-torch215.toml) (bitsandbytes on the graphics card, llama.cpp on the processor), and [comparisons/rewrite-cpu-runtimes-v3.toml](../comparisons/rewrite-cpu-runtimes-v3.toml) and [comparisons/rewrite-cpu-runtimes-v3-fp32.toml](../comparisons/rewrite-cpu-runtimes-v3-fp32.toml) (bitsandbytes on the processor). The three-passage averages come from the `responses.jsonl` of the same runs.
- Word checks of the llama.cpp rewrites: [comparisons/rewrite-llama-cpp-v3.toml](../comparisons/rewrite-llama-cpp-v3.toml) and [comparisons/rewrite-llama-cpp-v3-holdout.toml](../comparisons/rewrite-llama-cpp-v3-holdout.toml).
- The llama.cpp rewrites judged, and the speed of llama.cpp on the desktop's graphics card and on the MacBook: [comparisons/rewrite-keypoints-llama-cpp-v3-desktop.toml](../comparisons/rewrite-keypoints-llama-cpp-v3-desktop.toml) and [comparisons/rewrite-keypoints-llama-cpp-v3-macbook.toml](../comparisons/rewrite-keypoints-llama-cpp-v3-macbook.toml), with their `-holdout-desktop` and `-holdout-macbook` twins. The table combines their four `summary.csv` files.
- The judge's prompts and replies for those six runs: `results/judgments/rewrite/rewrite-keypoints-llama-cpp-v3/` and `…-holdout/`, prepared from [comparisons/rewrite-keypoints-llama-cpp-v3.toml](../comparisons/rewrite-keypoints-llama-cpp-v3.toml) and its `-holdout` twin. The counts of changed verdicts and of identical rewrites come from these records and from the runs' `responses.jsonl`.
- Not saved in the repository: the MacBook's bitsandbytes run, and every memory figure in [How fast does it run?](#how-fast-does-it-run).
