# OmniEvaluator
# Copyright (c) 2026-present NAVER Cloud Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

JUDGE_PROMPT = """[GOAL]
Given a **Question**, **Response**, and a natural-language **Checklist**, decide for each checklist item whether the Response **explicitly** satisfies it: **met = 1**, **not met = 0**. Final score = **(# met) / (total checklist items)**.

[INPUT]
[Question]
{QUESTION}

[Response]
{RESPONSE}

[Checklist]
{CHECKLIST}  ← JSON array or a plain list string. Treat each string as one criterion. Strip any leading numbering like “1.” or “2)”.

[DECISION RULES]
1. **Use only the Response text.** No outside knowledge/assumptions. If uncertain → 0.
2. **Explicitness (mentions/explains/indicates).**
   * 1: Clear, direct statement that fulfills the criterion.
   * 0.5: Indirect/implicit mention that likely implies fulfillment, but not explicit.
   * 0: Not mentioned or contradicted.
   
3. **“All / every / complete” requirements.**
   * 1: Explicitly states completeness (e.g., “all”, “every”, or equivalent).
   * 0.5: Strongly suggests near-completeness (“fill the nests” without “all”, “almost all”).
   * 0: No completeness requirement or states partial suffices.
   
4. **Method / Procedure (“explains how / method”).**
   * 1: Concrete, actionable steps or clear guidance.
   * 0.5: Vague or partial steps (general approach without specifics).
   * 0: No method/procedure provided.
   
5. **“Various / multiple types.”**
   * 1: Names **≥2 distinct, specific types**.
   * 0.5: Mentions variety without naming types, or names only **1** type.
   * 0: No indication of multiple types.
   
6. **Synonyms.** Accept unambiguous equivalents (e.g., “baby dragon” = “hatchling”).
   * 1: Unambiguous equivalence.
   * 0.5: Likely equivalent but slightly ambiguous.
   * 0: Ambiguous or different meaning.

[Evidence policy]
* For **met = 1 or 0.5**, include a **10–60 character direct quote** from the Response supporting the decision.
* For **met = 0 ** include a brief explanation why the response fails the given criteria.
* In the evidence block, list **evidence first**, then the explanation, then the met value.

[OUTPUT FORMAT — STRICT. NO PROSE OUTSIDE TAGS.] <evidence>
Item 1:
evidence: "…direct quote from Response…"
explanation: Briefly justify why criterion 1 earned 1/0.5/0 (reference rule numbers if helpful).
met: 0 | 0.5 | 1

Item 2:
evidence: "…"
explanation: …
met: 0 | 0.5 | 1

… (repeat for all checklist items, in order) </evidence>
<score>
K/N 
</score>

[NOTES]
* Output **only** the two tags above; no code fences, no extra text."""
