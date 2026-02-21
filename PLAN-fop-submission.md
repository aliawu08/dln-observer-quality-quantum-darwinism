# Implementation Plan: Foundations of Physics Submission Folder

**Prepared by:** Head of Research Advisory
**Date:** 2026-02-21
**Branch:** `claude/review-preprint-guidelines-bIS0O`

---

## 0. Strategic Framing

Foundations of Physics publishes work on *"the conceptual bases and fundamental theories of modern physics."* This audience cares about foundational clarity, interpretive precision, and conceptual novelty more than technical compression. The PRA version was trimmed for a technical audience that wants concise results; the FoP version should **restore conceptual depth** (closer to the preprint) while strictly conforming to Springer Nature formatting.

The key editorial pitch for FoP: this paper addresses a foundational gap — the observer is the least-modeled component in quantum Darwinism, despite being central to the entire interpretive program. The DLN connection, which was deliberately minimized for PRA, is *more appropriate* for FoP's interdisciplinary-foundations readership.

---

## 1. Folder Structure

Create `paper/fop/` mirroring the PRA pattern:

```
paper/fop/
├── paper4a_observer_quality_fop.tex          # Main manuscript (Springer sn-jnl class)
├── paper4a_observer_quality_major_revision.bib  # Bibliography (shared name, FoP-adapted)
├── cover_letter.tex                          # Cover letter to FoP editors
├── cover_letter.pdf                          # Compiled cover letter
├── sn-jnl.cls                               # Springer Nature journal class (if bundled)
├── sn-basic.bst                             # Springer Nature bibliography style
├── central_spin_redundancy_vs_time.pdf       # Fig 1 (copied from figures/)
├── inverted_sophistication_crossover.pdf     # Fig 2 (copied from figures/)
└── paper4a_observer_quality_fop.pdf          # Compiled PDF
```

**Rationale for figure selection:** The preprint uses 2 figures. The PRA version uses 4. For FoP, we recommend **3 figures**: the two from the preprint plus `robustness_decoherence_models.pdf` from the dynamical-redundancy script, which illustrates the inverted-sophistication decoherence-model comparison — a conceptually important point for a foundations audience. (The PRA's additional `central_spin_m_required_vs_time.pdf` and `central_spin_robustness_vs_p.pdf` are technical robustness checks more suited to PRA than FoP.)

---

## 2. Document Class and Template Adaptation

### 2.1 Switch from `article` to Springer Nature `sn-jnl`

The preprint uses `\documentclass[11pt]{article}` with custom geometry. The FoP version should use:

```latex
\documentclass[sn-basic]{sn-jnl}
```

This handles:
- Single-column layout with Springer formatting
- Proper `\title{}`, `\author[]{}`, `\affil[]{}` commands
- `\abstract{}` environment with correct formatting
- `\keywords{}` command
- Numbered citation style with square brackets
- "Fig." caption prefix (bold, abbreviated) automatically
- Declarations section formatting

**If the Springer template files are unavailable** in the build environment, fall back to:
```latex
\documentclass[11pt]{article}
```
with manual adjustments (numeric natbib, Springer-compatible bst, manual keywords block). The plan below handles both paths.

### 2.2 Package Changes from Preprint

| Preprint package | FoP action | Reason |
|---|---|---|
| `geometry` (1in margins) | Remove if using `sn-jnl`; keep if fallback `article` | Template handles margins |
| `natbib` (numbers, sort&compress) | Keep with `numbers` option | FoP requires numeric [1] citations |
| `enumitem` | Keep | Used for list formatting |
| `hyperref` | Keep but move to last loaded | Standard practice |
| `graphicx` | Keep | Figures |
| `booktabs` | Keep | Table formatting |

### 2.3 Macros

Start from the preprint macro set. Add any macros the PRA version introduced (none significant). Remove unused macros if any.

---

## 3. Content Strategy: What to Keep, Cut, and Add

This is the most consequential set of decisions. The preprint is ~1170 lines; the PRA version is ~614 lines. FoP should land at approximately **900–1000 lines** — fuller than PRA but tighter than the preprint.

### 3.1 Section-by-Section Plan

| Section | Preprint | PRA | FoP plan |
|---|---|---|---|
| **Introduction** | ~25 lines, includes "Scope" and "Why epsilon-SBS" paragraphs | ~20 lines, more compressed | **Restore preprint length.** FoP readers value motivation and positioning. Keep both "Scope" and "Why epsilon-SBS" paragraphs but convert `\paragraph{}` to inline bold text (avoids 4th heading level). |
| **Related work** (Sec 2) | Present as standalone section | Absent (folded into intro) | **Keep as standalone section.** FoP values explicit positioning against prior work. This is a strength of the preprint. |
| **Setup: SBS, epsilon-SBS, observer quality** (Sec 3) | Full section with SBS, epsilon-SBS, observer quality, DLN instantiation (Secs 3.1–3.3) | Compressed; DLN reduced to footnote + table | **Restore the full DLN instantiation** (Sec 3.3 of preprint), including the bipartite DAG proposition, decoder class definitions, strict nesting proof, tight exponent separation, and revision graph formalism. This is the paper's most conceptually distinctive contribution and aligns perfectly with FoP's mission. Remove the PRA's standing-assumptions block (useful for PRA's terse style but unnecessary when the text has room to be explicit). |
| **Notation table** | Absent | Present (Table I) | **Include.** It aids readability regardless of audience. Adapt from PRA's Table I. |
| **Chernoff decoding bounds** (Sec 4) | Full proofs inline | Theorems only; proofs in appendix | **Proofs inline** (FoP style is single-column and has space; appendix proofs are a PRA/letter convention). |
| **Heterogeneous fragments** (PRA Sec IV) | Absent | Present | **Include as a short remark** rather than a full section. The point (optimal subset selection for heterogeneous fragments) is relevant but not central. |
| **epsilon-SBS robustness** (Sec 5) | Full section with inline proof | Short section; proof in appendix | **Keep full section with inline proof.** |
| **Local vs collective decoding** (Sec 6) | Full section with Propositions 5–6, Corollary 1, Remark | Similar content, slightly compressed | **Keep preprint version.** The scope-clarification (Prop 5: no universal gap for pure hypotheses) is pedagogically important. |
| **Central-spin model** (Sec 7) | 1 subsection on limitations; 1 figure | 4 subsections, 3 figures, robustness checks, detailed limitations | **Use a middle ground.** Keep 2 figures (redundancy + robustness-vs-p). Add the scope/limitations subsection from PRA (Sec VI.D) — it's well-written and anticipates reviewer objections. Skip the heterogeneous-access detail (PRA Sec VI.C). |
| **DLN-series connection** (Sec 8) | Standalone section | Absent (footnote only) | **Keep as standalone section.** For FoP, this explicit bridge between quantum foundations and cognitive/information-processing frameworks is a selling point, not a distraction. |
| **Inverted sophistication** (Sec 9) | Full section with 3 decoherence models | Subsection within central-spin | **Keep as standalone section** (preprint structure). The conceptual point — that "more sophisticated" observers can do worse — is exactly the kind of foundational insight FoP publishes. Include the `robustness_decoherence_models.pdf` figure here. |
| **Discussion** (Sec 10) | "Tightness, limitations, and relation to prior work" | Absent (folded into conclusion) | **Keep as standalone section.** FoP expects substantive discussion. |
| **Conclusion** (Sec 11) | ~8 lines | ~12 lines (PRA packs more into conclusion) | **Hybrid.** Use preprint structure but incorporate PRA's sharper scope-limitation language. |
| **Acknowledgments** | Absent | Present | **Add.** Required by FoP. |
| **Appendix** | Absent | Present (all proofs) | **Omit.** Proofs are inline. |

### 3.2 Content to Add (Not in Either Version)

These are **required by FoP guidelines** and currently missing from both versions:

1. **Keywords** (4–6, after abstract)
2. **Declarations section** (before references):
   - Competing Interests
   - Funding
   - Data Availability
3. **Acknowledgments section**
4. **AI/LLM disclosure** (if applicable; in Methods or equivalent)

---

## 4. Specific Formatting Fixes

### 4.1 Title
Shorten to fit FoP's "concise and informative" requirement:

**Proposed:**
> "Observer Quality as a Resource Variable in Quantum Darwinism"

The full subtitle content moves to the abstract.

### 4.2 Author Block
Add city and country:

```latex
\author{Alia Wu}
\affil{Risk Efficacy \& Redline Rising, New York, NY, USA}
\email{wut08@nyu.edu}
\orcid{0009-0005-4424-102X}
```

(Confirm actual city/country with the author.)

### 4.3 Abstract
Rewrite to 150–250 words. Target: **230 words**. Strategy:
- Keep the first two sentences (QD/SBS framing + observer gap).
- State the three main technical contributions (i–iii) concisely.
- Mention the central-spin example in one sentence.
- Mention the DLN connection and inverted sophistication in one sentence each.
- Eliminate all undefined abbreviations on first use (spell out CPTP, SBS, etc.).

### 4.4 Keywords
```latex
\keywords{quantum Darwinism \and spectrum broadcast structure \and observer quality \and quantum Chernoff bound \and quantum-to-classical transition \and decoherence}
```

### 4.5 Heading Levels
Replace all `\paragraph{}` with either:
- `\subsubsection{}` (if the content warrants a heading), or
- Inline bold text: `\textbf{Topic.}` (if it's a short labeled paragraph)

Audit: the preprint has ~6 `\paragraph{}` uses. Most should become inline bold.

### 4.6 Citation Style
- Use `\bibliographystyle{sn-basic}` (Springer Nature) or `spbasic` as fallback
- Replace all `\citet{...}` with `\citep{...}` or equivalent — FoP uses `[1]` style, not `Author [1]` style
- Exception: constructions like "Zurek [3] showed..." are acceptable in FoP's numeric style, but should be used sparingly

### 4.7 Figure Captions
- Remove all trailing periods from captions
- Verify "Fig." prefix renders correctly (Springer template handles this; manual fix if using `article` class)

### 4.8 Abbreviations
Define at first use in body text:
- **CPTP** (completely positive trace-preserving)
- **DAG** (directed acyclic graph)
- **POVM** (positive operator-valued measure)
- **i.i.d.** (independent and identically distributed)
- **SBS** (already defined in preprint Sec 3.1 but must also be defined in abstract)

### 4.9 Bibliography
- Start from the preprint `.bib` file (382 entries, superset)
- Add `WuZenodo2026` and `Unden2019` entries from the PRA bib (needed for acknowledgments/experimental references)
- Verify all DOIs are present
- Journal name abbreviations: handled by the `.bst` file (Springer's `sn-basic.bst` abbreviates automatically)

---

## 5. Cover Letter

Write a new cover letter addressed to the FoP editorial office. Key differences from the PRA cover letter:

1. **Pitch angle:** Emphasize the foundational/conceptual contribution (observer as under-modeled component in QD; DLN bridge between quantum foundations and cognitive science) rather than technical novelty alone.
2. **Why FoP:** Explicitly state why Foundations of Physics is the right venue — the paper addresses a conceptual gap at the intersection of decoherence theory, quantum Darwinism, and the observer problem.
3. **Suggested reviewers:** Keep Korbicz and Roszak (SBS/pure-dephasing experts). Replace Le with someone more aligned with FoP's foundations audience, e.g.:
   - **Maximilian Schlosshauer** (University of Portland) — decoherence and quantum-to-classical transition
   - **Wojciech H. Zurek** (LANL) — originator of quantum Darwinism (senior choice; may or may not be appropriate depending on perceived conflict)
   - **Sebastian Deffner** (UMBC) — quantum thermodynamics and operational approaches to QD
4. **Simultaneous-submission statement:** Explicitly state the paper is not under review elsewhere.
5. **DLN framing:** Briefly note the companion preprint but emphasize self-containedness.

---

## 6. Makefile Updates

Add a `fop` target:

```makefile
fop:
	cd paper/fop && pdflatex -interaction=nonstopmode paper4a_observer_quality_fop.tex
	cd paper/fop && bibtex paper4a_observer_quality_fop
	cd paper/fop && pdflatex -interaction=nonstopmode paper4a_observer_quality_fop.tex
	cd paper/fop && pdflatex -interaction=nonstopmode paper4a_observer_quality_fop.tex
```

Add FoP figures to the `figs` target:

```makefile
FOP_FIGS := central_spin_redundancy_vs_time.pdf \
            inverted_sophistication_crossover.pdf \
            robustness_decoherence_models.pdf

# In the figs target, add:
$(foreach f,$(FOP_FIGS),cp figures/$(f) paper/fop/;)
```

Add `fop` to `.PHONY` and the `clean` target.

---

## 7. Implementation Order

Execute in this sequence (dependencies noted):

| Step | Task | Depends on | Est. complexity |
|---|---|---|---|
| **7.1** | Create `paper/fop/` directory | — | Trivial |
| **7.2** | Obtain/create Springer Nature template files (`sn-jnl.cls`, `sn-basic.bst`) or confirm fallback strategy | — | Medium (may need download) |
| **7.3** | Create `paper4a_observer_quality_fop.tex` — adapt from preprint `.tex` per Sec 3 content strategy | 7.1, 7.2 | **High** (largest task) |
| **7.4** | Create FoP-adapted `.bib` file | 7.1 | Low (merge preprint + PRA extras) |
| **7.5** | Rewrite abstract (150–250 words) | 7.3 | Medium |
| **7.6** | Add keywords, Declarations, Acknowledgments | 7.3 | Low |
| **7.7** | Fix all `\paragraph{}` → inline bold or `\subsubsection{}` | 7.3 | Low |
| **7.8** | Fix citation commands (`\citet` → `\citep` where needed) | 7.3 | Low |
| **7.9** | Fix figure captions (remove trailing periods) | 7.3 | Low |
| **7.10** | Define all abbreviations at first use | 7.3 | Low |
| **7.11** | Write cover letter (`cover_letter.tex`) | 7.1 | Medium |
| **7.12** | Copy figure PDFs to `paper/fop/` | 7.1 | Trivial |
| **7.13** | Update `Makefile` with `fop` target and figure copies | 7.1, 7.12 | Low |
| **7.14** | Test build: `make fop` | 7.3–7.13 | Medium (debug cycle) |
| **7.15** | Verify compiled PDF against checklist in `REVIEW-foundations-of-physics-guidelines.md` | 7.14 | Medium |

**Critical path:** 7.2 → 7.3 → 7.5–7.10 → 7.14 → 7.15

Steps 7.4, 7.11, 7.12, 7.13 can be done in parallel with the main .tex work.

---

## 8. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Springer `sn-jnl.cls` not available in build environment | Build fails | Fallback: use `article` class with manual Springer-compatible formatting. Less polished but acceptable for initial submission. |
| Abstract reduction to ≤250 words loses key content | Reviewers miss contributions | Draft two versions: a 230-word version (aggressive trim) and a 245-word version (minimal trim). Choose the one that preserves the most clarity. |
| DLN self-citation count (~11) flagged by reviewers | Reviewer concern about self-promotion | Reduce to ~7–8 citations by consolidating references (e.g., cite once for the framework, not separately for each sub-result). Add a sentence in Sec 8 explicitly stating the DLN connection is offered as structural context and the quantitative results are self-contained. |
| `WuDLNCompression2026preprint` still a preprint at submission time | May violate FoP reference policy | Check if it has been accepted/published. If still a preprint, add a note in the reference: "preprint, under review" and be prepared for an editor query. FoP's physics community generally tolerates arXiv/preprint citations. |
| Simultaneous submission concern (PRA version exists) | Ethical violation | Confirm with author that PRA submission is withdrawn or was never submitted before submitting to FoP. Add explicit statement in cover letter. |

---

## 9. Deliverables Checklist

Upon completion, the following should exist and be verified:

- [ ] `paper/fop/paper4a_observer_quality_fop.tex` — compiles without errors
- [ ] `paper/fop/paper4a_observer_quality_major_revision.bib` — complete bibliography
- [ ] `paper/fop/cover_letter.tex` + `.pdf` — addressed to FoP editors
- [ ] `paper/fop/*.pdf` — all required figure files copied
- [ ] `paper/fop/paper4a_observer_quality_fop.pdf` — compiled manuscript
- [ ] `Makefile` — updated with `fop` target and figure copies
- [ ] Abstract ≤ 250 words with all abbreviations defined
- [ ] Keywords present (4–6)
- [ ] Declarations section present (Competing Interests, Funding, Data Availability)
- [ ] Acknowledgments section present
- [ ] All headings ≤ 3 levels deep
- [ ] Numeric `[1]` citation style throughout
- [ ] No trailing periods on figure captions
- [ ] All abbreviations defined at first use
- [ ] Affiliation includes city and country
