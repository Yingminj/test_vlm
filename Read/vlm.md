I've read the full paper including the appendices. Here's a detailed analysis structured for your presentation, covering all five perspectives plus the training-data and training-order details you specifically asked about.

---

# R³-Streaming: An Efficient Streaming Video Understanding Framework with Agentic Control

**Authors:** Jinming Liu, Jianguo Huang et al. (SJTU / Eastern Institute of Technology / Microsoft Research Asia) · arXiv 2605.17921v2

---

## 1. Background & Motivation — *Why this work?*

**The setting.** Offline video-LLMs (captioning, summarization, long-context QA) work well because they see the *whole* clip at once. But **streaming** video is different: frames arrive continuously, and every decision must be made *now*, under a strict latency budget, without seeing the future.

**The pain points / gaps they identify:**

- **Static, single-point optimization.** Existing streaming methods each optimize *one* piece of the pipeline — only response triggering, *or* only token/frame compression. Memory retention, response timing, and reasoning depth are never *dynamically coordinated*.
- **The forced trade-off.** Fast small models fail on complex queries; "always-on" heavy models violate real-time constraints *and* over-think simple queries. Prior decision-reaction frameworks rely on a single reaction model → a **sub-optimal efficiency-performance trade-off**.
- **No adaptation to information density.** Real streams have varying informational density; a fixed strategy can't adapt.

**Two empirical findings that motivate the design (Sec. 3, Fig. 1–2):**

| Finding | Evidence | Implication |
|---|---|---|
| **F1: Important signal is strongly recent-focused; historical tokens are redundant and even *misleading*.** | Deletion analysis on OVO-Bench (measured by Jensen-Shannon Divergence of next-token distribution): historical tokens receive *most* of the visual attention, yet removing them barely shifts the output — removing *nearby* tokens causes ~1.7× larger shift. Aggressively compressing history (Nearby=1.0, Historical=0.01) *raises* StreamingBench by **+2.4** (71.0 vs no compression). | → Motivates **Remember** (Active Forgetting): keep recent context at high fidelity, aggressively forget old. |
| **F2: Model-scale gains are non-monotonic across streaming tasks.** | Qwen2.5-VL-**3B** beats the **7B** on Realtime/Forward tasks; Qwen3-VL-4B-Thinking ≈ 8B-Thinking. "Always-on" heavy inference is both expensive *and* sometimes worse. | → Motivates **Reason** (Adaptive Thinking): route only hard queries to the slow model. |

> **Slide takeaway:** They reframe streaming understanding from a *passive feed-forward* task into an **agentic control problem** — the agent manages its own state and chooses its own compute path.

---

## 2. Core Innovation — *What's new?*

**R³-Streaming = Remember, Respond, Reason** — a **cascaded control framework** that decomposes each streaming step into three sequentially-coupled decisions, so each downstream decision builds on a progressively refined information state:

1. **Remember (Active Forgetting)** — a *training-free*, age-aware memory compressor. Recent frames kept at high fidelity, stale frames consolidated into compact slots.
2. **Respond (Proactive Response)** — a lightweight *readiness head* that emits `<Routine>` to **defer** answering when evidence is insufficient (avoids premature hallucination).
3. **Reason (Adaptive Thinking)** — a *routing policy*: the fast model either answers directly (`<Answer>`) or emits `<Escalate>` to invoke a heavier reasoning/thinking model.

**Two headline technical contributions:**

- **Age-aware forgetting policy** — the empirical insight that *aggressively compressing history improves accuracy* (not just saves compute), turned into a dual-zone memory schedule.
- **TB-GRPO (Target-Balanced GRPO)** — a reinforcement-learning objective that **stabilizes the binary routing policy** and keeps the escalation rate inside a *deployable* compute budget. This is the key novelty: vanilla GRPO and AutoThink collapse to "always escalate"; TB-GRPO adds explicit **target-band control** over escalation frequency.

**Action space:** `A = {<Answer>, <Escalate>, <Routine>}`.

> **Why it matters:** On StreamingBench, the adaptive router *outperforms* direct slow-only inference — so the gain is **not** a simple fixed fast/slow blend; agentic control genuinely picks the better action per query.

---

## 3. Methodology — *How is it built?*

### 3.1 Problem formulation
At step *t*: observe stream history `x₁:ₜ`, receive query `qₜ`, maintain memory `Mₜ`, then a **cascade** of decisions. Because each decision consumes the previous one's output, **errors compound** (noisy memory → bad readiness → unnecessary escalation) — which is exactly why they coordinate the three jointly.

### 3.2 Remember (Active Forgetting) — *training-free*
Partition history into a **nearby** window (size *W*) and a **historical** zone, compress each with different thresholds:

$$M_t = \text{Compress}(x_{t-W+1:t},\ \tau_{near}) \cup \text{Compress}(x_{1:t-W},\ \tau_{hist})$$

with **τ_near ≫ τ_hist** (near = fine-grained tokens; far = compact episodic slots).
- Compression operator: **DTD** (from TimeChat-Online). Shown robust to operator choice (DivPrune/VisionZip/AvgPooling all work) — the *age-aware policy* matters more than the operator.
- Default (streaming): Nearby threshold **1.0** (no compression), Historical **0.01**, Nearby window **3 frames** → **95–96% token drop**.
- Offline long-video: Historical reset to **0.5** (~45% drop) — a deterministic *scenario toggle*, not a per-video search, since deployment knows a priori whether the stream is live or pre-recorded.

### 3.3 Respond (Proactive Response)
A readiness head *h* estimates answerability:

$$p_{ready} = h(q_t, M_t), \qquad a_t = \begin{cases} \text{emit } \texttt{<Routine>}, & p_{ready} < 0.5 \\ \text{continue to Reason}, & \text{otherwise} \end{cases}$$

### 3.4 Reason (Adaptive Thinking) — the hard part
Binary routing: answer directly vs `<Escalate>`. Learned in **two stages** (SFT cold-start → TB-GRPO; details in §5 below).

**TB-GRPO mechanics.** For a query, the policy samples a group `{yᵢ}`; `eᵢ = 1[yᵢ = <Escalate>]`, `cᵢ` = correctness, `ρ = (1/G)Σeᵢ` = group escalation ratio.

*Base reward* (encodes "answer when capable, escalate when necessary"):

$$r^{naive}_i = \begin{cases} 2 & e_i{=}0, c_i{=}1 \;\text{(correct direct answer — best, lowest latency)}\\ -1 & e_i{=}0, c_i{=}0\\ 1 & e_i{=}1, c_i{=}1\;\text{(correct but slow)}\\ 0 & e_i{=}1, c_i{=}0 \end{cases}$$

*Target-band penalties* around operating point (η, γ):

$$\delta_{esc} = \text{clip}(\rho - (\eta+\gamma), 0, 1), \qquad \delta_{ans} = \text{clip}((\eta-\gamma) - \rho, 0, 1)$$

- `ρ > η+γ` (over-escalating) → δ_esc>0 suppresses escalation reward
- `ρ < η−γ` (under-escalating) → δ_ans>0 penalizes direct answers
- inside the band → both zero (proportional/feedback control on routing frequency)

These modulate the base reward `rᵢ`, then standard **group-normalized advantages** `Aᵢ = (rᵢ − r̄)/(std+ε)` with a **clipped GRPO + KL** objective:

$$\mathcal{L}_{TB\text{-}GRPO} = \mathbb{E}[\min(w_i A_i, \text{clip}(w_i, 1-\epsilon_c, 1+\epsilon_c)A_i)] - \beta_{KL} D_{KL}(\pi_\theta \| \pi_{ref})$$

Setting: **η = 0.3, γ = 0.2**. The (η, γ) pair is a direct, tunable control knob for the compute budget.

> **Training-dynamics evidence (Fig. 6):** Vanilla GRPO collapses to ρ=1.0 by step 40; AutoThink avoids early collapse but settles high; **TB-GRPO converges to a lower, stable ratio *and* a higher reward.**

---

## 4. Experiments — *Does it work?*

**Setup.** Fast = Qwen2.5-VL-3B/7B; Slow = Qwen3-VL-4B/8B-Thinking or Qwen2.5-VL-32B. Notation `R3-Streaming-[Fast]|[Slow]`.

**Main streaming results (SOTA among streaming MLLMs):**

| Benchmark | Best config | Score | Note |
|---|---|---|---|
| **OVO-Bench** | 7B \| 4B-Thinking | **57.92** | 96% token reduction; beats Streamo-7B (55.61), StreamAgent-7B (49.4), TimeChat-Online (45.6) |
| **StreamingBench** | 7B \| 4B-Thinking | **76.36** | 95% token reduction; beats StreamAgent (74.28), Dispider (67.63), even GPT-4o (73.28) |

Notably it **surpasses both** its own slow baseline (Qwen3-VL-4B-Thinking, 57.74/73.16) and fast baseline (Qwen2.5-VL-7B) → gain isn't a fixed fast/slow blend.

**Generalization to offline long video (Table 3):** With Historical=0.5, R3-Streaming-3B|4B = **MLVU 70.6 / Video-MME 65.5**, beating AdaReTaKe, VideoAgent, TimeChat-Online, StreamAgent.

**Key ablations:**

- **Modules are complementary (Table 4):** baseline 3B → +Remember (52.8) → +Reason (55.9) → **+Both = 56.6** on OVO; same trend on StreamingBench.
- **Remember (Table 5):** age-aware DTD = **75.90** & 95% drop vs DivPrune/VisionZip/DTD-alone (65–68%). Gain comes from the *policy*, not the operator.
- **Respond (Table 6):** dedicated readiness head = **0.328** proactive output vs 0.216 (3B) / 0.204 (7B) — learned, not from scale.
- **Reason routing (Table 7):** SFT-only & Vanilla GRPO → 100% escalation (collapse); AutoThink → 53.2% escalate, 70.0 acc; **TB-GRPO → 74.36 acc at just 24.0% escalation.**
- **(η, γ) sweep (Table 8):** explicit accuracy↔escalation control, e.g. η=0.3,γ=0.2 → 74.4 acc / 24% escalate.
- **Efficiency (Fig. 7):** adaptive routing beats slow-only across *all* slow models at far lower per-frame latency.
- **Memory-routing synergy (Appendix D.3):** restoring Nearby=1.0 raises Overall accuracy 64.72→73.64% *while* dropping escalation 37.04→24.04% — better recent context makes the fast path self-sufficient. Routing adapts to difficulty: Object Perception ~20% escalation; Counting 49–72%, Prospective Reasoning >65%.

---

## 5. Training Data Construction & Training Order *(your specific ask)*

**Overall training order across modules:**

```
① Remember  → training-free (NO parameters; just a config toggle)
② Reason Stage-1: SFT cold-start  (teach the routing *format*)
③ Reason Stage-2: TB-GRPO RL      (teach routing *decision quality*)
④ Respond: readiness-head SFT     (trained LAST, after ②+③ finish)
```
Respond is deliberately trained **last**, with the **fast VLM frozen** — only the lightweight readiness head is updated — so it doesn't perturb the already-optimized fast/slow router.

### Data construction per module

**② Reason — SFT cold-start data (Fig. 4, Appendix D.1).** Built from **TimeChat-Online-139K**.
- For each query, the fast model samples **K = 4** responses.
- **Open-ended questions:** scored by an external LLM (**Qwen3-14B**) comparing response vs ground truth on a **5-point scale**.
- **Objective questions:** binary exact-correctness `s ∈ {0,1}`.
- Aggregate `s̄ = (1/K)Σs`, then label: `s̄ ≥ T → <Answer>`, else `<Escalate>`. Threshold **T = 2.5**.
- Fast model is fine-tuned on these mixed targets — this fixes *format* only ("speak the routing language"), not decision quality.

**③ Reason — TB-GRPO RL data.** Also from TimeChat-Online-139K; reward computed online from correctness + the target-band rule above (η=0.3, γ=0.2). This refines *when* to escalate.

**④ Respond — readiness SFT data (Appendix C).** Built from **TimeChat-Online-139K + COIN**, using **decision-boundary-focused hard-mining** rather than random sampling:
- Identify the exact **clue timestamp t_c** where the question first becomes answerable.
- **Unready → `<Routine>`:** the 3 frames *just before* the clue, `{t_c−3, t_c−2, t_c−1}`.
- **Ready → proceed to Reason:** the clue frame + next 2, `{t_c, t_c+1, t_c+2}`.
- Boundaries clipped to valid range; frames outside this narrow window are *deliberately omitted*. This concentrates supervision on the just-before/just-after decision boundary → learns to avoid premature hallucination while minimizing reaction latency.

---

## 6. Conclusion & Limitations

**Insights / contributions:**
- Streaming understanding is best framed as **cascaded agentic control**, not feed-forward inference.
- **Forgetting history is actively beneficial** (noise removal), not just a compression compromise.
- **Selective reasoning** (TB-GRPO) beats always-on heavy models on both accuracy *and* latency, with a tunable budget knob.
- SOTA streaming results + strong offline generalization + 95–96% token reduction.

**Limitations (stated + my read):**
- **Worst-case latency spikes:** TB-GRPO only constrains the *average* escalation ratio. During prolonged extreme-information-density segments, even fast models struggle and escalations may cluster → bursts can still exceed the budget. Current benchmarks only measure *averages*; the authors call for **spike-aware benchmarks**.
- *(Additional angles you can raise in discussion):* Respond uses a fixed 0.5 threshold; SFT labels depend on an LLM judge (Qwen3-14B) quality; the online/offline Historical threshold is a manual toggle rather than learned; routing is binary (no multi-tier "medium" model); evaluated only on Qwen-family backbones.

**Relevance to your work:** the Remember module is essentially an *object/evidence-permanence memory* problem — the JSD-based finding that *recent* tokens dominate while history misleads attention is a useful counterpoint to your persistent-object-memory direction, where you specifically *want* to retain occluded-object state across long gaps. Their age-aware forgetting would discard exactly the evidence your project needs to keep, which is an interesting tension worth a slide if this feeds your VLM object-state memory project.

---

### Suggested slide deck structure (12–14 slides)
1. Title + one-line thesis 2. The streaming problem & the forced trade-off 3. Finding 1 (recent-focused attention, JSD) 4. Finding 2 (non-monotonic scaling) 5. R³ framework overview (Remember/Respond/Reason cascade) 6. Remember: dual-zone Active Forgetting + Eq.(1) 7. Respond: readiness head + Eq.(2–3) 8. Reason: routing + SFT data pipeline 9. TB-GRPO reward + target band (Fig. 5) 10. Training dynamics (Fig. 6) 11. Main results (OVO + StreamingBench tables) 12. Ablations (4 modules) + efficiency Fig. 7 13. Training data & order summary 14. Conclusion / limitations / future (spike-aware).

---

Would you like me to **generate the actual `.pptx` file** from this analysis? If so, tell me:
- **Language** — English, or Chinese (中文)?
- **Depth** — concise (~12 slides, key points) or detailed (~16–18 slides with full formulas/tables)?

I can also embed the key equations and a redrawn pipeline diagram. Just say the word and I'll build it.