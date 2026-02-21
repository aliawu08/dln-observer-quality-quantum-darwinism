# Review: Preprint vs. Foundations of Physics Submission Guidelines

**Manuscript:** "Observer Quality as a Resource Variable in Quantum Darwinism: Optimal Decoding, epsilon-Approximate Spectrum Broadcast Structure, and a Central-Spin Worked Example"
**Author:** Alia Wu
**Target journal:** Foundations of Physics (Springer Nature)
**Date of review:** 2026-02-21

---

## Summary

This review evaluates the manuscript against each section of the *Foundations of Physics* submission guidelines (Springer Nature). Items are marked PASS, NEEDS ATTENTION, or FAIL. An action-item checklist is provided at the end.

---

## 1. Manuscript Submission (General)

| Requirement | Status | Notes |
|---|---|---|
| Not published elsewhere | PASS | Preprint on PhilSci-Archive / Zenodo; preprints are generally acceptable |
| Not under simultaneous consideration | NEEDS ATTENTION | The repository also contains a PRA (Physical Review A) formatted submission under `paper/pra/`. If the paper is concurrently submitted to PRA and FoP, this violates the simultaneous-submission policy. The author must confirm the paper is submitted to only one journal at a time. |
| Publication approved by all co-authors | PASS | Single author |
| Editable source files provided | PASS | LaTeX `.tex` source, `.bib`, and all figures are available |

**Action items:**
- [ ] Confirm the paper is NOT simultaneously submitted to Physical Review A (or any other journal). If the PRA version exists only as a prepared alternative, this is fine, but it must not be under active review elsewhere.

---

## 2. Title Page

### 2.1 Title

| Requirement | Status | Notes |
|---|---|---|
| Concise and informative | NEEDS ATTENTION | The current title is long (3 lines in LaTeX). FoP prefers concise titles. Consider shortening. |

Current title:
> "Observer Quality as a Resource Variable in Quantum Darwinism: Optimal Decoding, epsilon-Approximate Spectrum Broadcast Structure, and a Central-Spin Worked Example"

**Suggested shortened version:**
> "Observer Quality as a Resource Variable in Quantum Darwinism"

The subtitle content (Chernoff decoding, epsilon-SBS, central-spin example) can be conveyed in the abstract and keywords instead.

### 2.2 Author Information

| Requirement | Status | Notes |
|---|---|---|
| Author name | PASS | Alia Wu |
| Affiliation (institution, department, city, state, country) | NEEDS ATTENTION | Currently listed as "Risk Efficacy & Redline Rising" with no city, state, or country. FoP requires institution, city, and country at minimum. If unaffiliated, city and country of residence are still required. |
| Corresponding author email | PASS | wut08@nyu.edu |
| ORCID | PASS | 0009-0005-4424-102X provided |

**Action items:**
- [ ] Add city and country to the affiliation. If "Risk Efficacy & Redline Rising" is an independent organization, provide its location (e.g., "Risk Efficacy & Redline Rising, New York, NY, USA"). If unaffiliated, provide city and country of residence.
- [ ] Consider whether the NYU email implies an NYU affiliation that should be listed.

### 2.3 LLM / AI Tool Disclosure

| Requirement | Status | Notes |
|---|---|---|
| AI tool usage documented in Methods | NEEDS ATTENTION | FoP requires that use of LLMs (beyond copy editing) be documented. If any LLM tools were used in drafting, analysis, or code generation, this must be disclosed. If not used, no disclosure is needed, but given the current landscape, a brief statement may be prudent. |

### 2.4 Abstract

| Requirement | Status | Notes |
|---|---|---|
| 150--250 words | NEEDS ATTENTION | The abstract is approximately 280 words (by rough count), exceeding the 250-word maximum. |
| No undefined abbreviations | NEEDS ATTENTION | Several abbreviations are used without definition in the abstract: "QD" (introduced but then "QD/SBS" is used without "SBS" being defined first), "SBS" (introduced mid-abstract), "CPTP" (not defined), "DLN" (defined only as "Dot--Linear--Network" in parentheses but may need more context), "i.i.d." (not defined). |
| No unspecified references | PASS | No references in the abstract |

**Action items:**
- [ ] Reduce the abstract to <= 250 words. The current abstract tries to cover all contributions; consider trimming the DLN/inverted-sophistication paragraph.
- [ ] Remove or define all abbreviations: spell out "CPTP" (completely positive trace-preserving) on first use, spell out "SBS" before using it, define or avoid "i.i.d."

### 2.5 Keywords

| Requirement | Status | Notes |
|---|---|---|
| 4--6 keywords provided | FAIL | No keywords are present in the manuscript. |

**Action items:**
- [ ] Add 4--6 keywords. Suggested: `quantum Darwinism`, `spectrum broadcast structure`, `observer quality`, `quantum Chernoff bound`, `quantum-to-classical transition`, `decoherence`.

### 2.6 Statements and Declarations

| Requirement | Status | Notes |
|---|---|---|
| Competing Interests declaration | FAIL | No competing interests statement is present. FoP requires this. |
| Funding declaration | FAIL | No funding statement is present. |
| Data Availability statement | FAIL | Not present (though the repository link partially addresses this). |
| Author Contributions | PASS | Single author; no statement needed. |

**Action items:**
- [ ] Add a "Declarations" section before the references containing:
  - **Competing Interests:** e.g., "The author has no relevant financial or non-financial interests to disclose."
  - **Funding:** e.g., "No funding was received for conducting this study." (or disclose any funding)
  - **Data Availability:** e.g., "All data and code supporting the findings of this study are openly available at https://github.com/aliawu08/observer-quality-quantum-darwinism (DOI: 10.5281/zenodo.18610548)."

---

## 3. Text Formatting

### 3.1 Document Class and Template

| Requirement | Status | Notes |
|---|---|---|
| LaTeX submission | PASS | LaTeX source provided |
| Springer Nature LaTeX template | NEEDS ATTENTION | The manuscript uses `\documentclass[11pt]{article}` with custom geometry, not the Springer Nature LaTeX template (`sn-jnl.cls`). FoP recommends (though does not strictly require) using the Springer Nature template. |

**Action items:**
- [ ] Consider reformatting using the Springer Nature LaTeX template (`sn-jnl.cls`) from https://www.springernature.com/gp/authors/campaigns/latex-author-support. This handles styling, section numbering, and metadata formatting automatically. Not strictly required, but recommended and may speed up acceptance/production.

### 3.2 Headings

| Requirement | Status | Notes |
|---|---|---|
| Decimal heading system, max 3 levels | NEEDS ATTENTION | The manuscript uses `\section`, `\subsection`, `\subsubsection`, and `\paragraph`. The `\paragraph` level is a 4th level. FoP allows at most 3 levels. |

**Action items:**
- [ ] Audit `\paragraph{}` usage throughout. Some can be promoted to `\subsubsection` or integrated into the text without a heading. For example, "Scope of this paper" and "Why an epsilon-SBS error model" in Sec. 1 could be folded into the introduction text.

### 3.3 Abbreviations

| Requirement | Status | Notes |
|---|---|---|
| Defined at first mention | NEEDS ATTENTION | Most abbreviations are defined, but "CPTP" appears in Definitions/Theorems without being spelled out at first mention in the main text. "i.i.d." is used without definition. "DAG" is used without being spelled out before its first appearance in a Proposition. |

**Action items:**
- [ ] Define "CPTP" (completely positive trace-preserving) at first use in the body text (around Sec. 3.2, Definition 3).
- [ ] Define "DAG" (directed acyclic graph) at first use (Sec. 3.3, Proposition 1 header).
- [ ] Define "i.i.d." (independent and identically distributed) at first use (Sec. 1 or Sec. 3.1).
- [ ] Define "POVM" (positive operator-valued measure) at first use in the main text.

### 3.4 Footnotes

| Requirement | Status | Notes |
|---|---|---|
| Use footnotes instead of endnotes | PASS | No endnotes used |
| Footnotes numbered consecutively | PASS | No footnotes present (acceptable) |

### 3.5 Acknowledgments

| Requirement | Status | Notes |
|---|---|---|
| Acknowledgments in separate section | FAIL | No Acknowledgments section. FoP requires this on the title page. Even if there is nothing to acknowledge, consider including a brief section. |

**Action items:**
- [ ] Add an Acknowledgments section (can be brief or state that there are none to declare).

---

## 4. References

### 4.1 Citation Style

| Requirement | Status | Notes |
|---|---|---|
| Numbered references in square brackets | NEEDS ATTENTION | The manuscript uses `natbib` with author-year style citations (e.g., `\citep{Zurek2009}` renders as "(Zurek, 2009)" or similar). FoP requires **numbered citations in square brackets** (e.g., [1], [2], [3]). |

**Action items:**
- [ ] Switch to numeric citation style. Change `\bibliographystyle{unsrtnat}` to a numeric style, or use `\usepackage[numbers,sort&compress]{natbib}` (already loaded, but verify the `.bst` file produces numbered output). The current `unsrtnat` with `numbers` option should produce numeric output -- verify the compiled PDF uses [1], [2], ... format.
- [ ] Change `\citet{...}` calls to `\citep{...}` or equivalent to get bracketed numbers in text. Review all `\citet` usages -- FoP style calls for "[3]" not "Zurek [3]".

### 4.2 Reference List Format

| Requirement | Status | Notes |
|---|---|---|
| Only cited works included | PASS | Bibliography appears to match citations |
| DOIs as full links | NEEDS ATTENTION | The `.bib` file includes `doi` fields but they are not formatted as full DOI links (e.g., `https://doi.org/10.1038/nphys1202`). FoP requests full DOI links in the reference list. |
| Journal names abbreviated per ISSN LTWA | NEEDS ATTENTION | Some entries use full journal names (e.g., "Physical Review Letters", "Nature Physics"). FoP guidelines say to use standard ISSN LTWA abbreviations (e.g., "Phys. Rev. Lett.", "Nat. Phys."). |

**Action items:**
- [ ] Verify DOIs render as full `https://doi.org/...` links in the compiled reference list. If not, adjust the bibliography style or add `\usepackage{doi}`.
- [ ] Standardize journal name abbreviations in the `.bib` file per ISSN LTWA, or use a `.bst` file that handles abbreviation automatically.

### 4.3 Reference Format

| Requirement | Status | Notes |
|---|---|---|
| FoP reference style followed | NEEDS ATTENTION | FoP uses a specific format: `Author(s): Title. Journal vol, pages (year)`. The current `unsrtnat` style may not match exactly. Verify against the example format in the guidelines. |

**Action items:**
- [ ] Compare compiled reference list format against FoP's examples. Consider using Springer's own `.bst` file (e.g., `spmpsci.bst` or `spbasic.bst`) for correct formatting.

---

## 5. Tables

| Requirement | Status | Notes |
|---|---|---|
| Tables numbered with Arabic numerals | PASS | Table 1 uses Arabic numeral |
| Cited in consecutive order | PASS | Table 1 is cited in text |
| Table caption provided | PASS | Caption present |
| Previously published material cited | N/A | Original content |

---

## 6. Artwork and Illustrations

### 6.1 Figure Format

| Requirement | Status | Notes |
|---|---|---|
| Electronic submission | PASS | PDF figure files provided |
| Appropriate format (EPS/TIFF preferred for vector/halftone) | NEEDS ATTENTION | Figures are in PDF format, which is acceptable but EPS is preferred for vector graphics. |
| Figure naming convention (Fig1.eps, etc.) | NEEDS ATTENTION | Figures are named descriptively (e.g., `central_spin_redundancy_vs_time.pdf`) rather than `Fig1.pdf`, `Fig2.pdf`, etc. |

### 6.2 Figure Captions

| Requirement | Status | Notes |
|---|---|---|
| Concise captions describing content | PASS | Captions are descriptive |
| Captions in text file, not figure file | PASS | Captions are in the `.tex` file |
| "Fig." in bold followed by bold number | NEEDS ATTENTION | LaTeX `\caption{}` will produce "Figure 1:" by default. FoP requires "**Fig. 1**" (bold, abbreviated). This is handled by the document class -- switching to Springer's template would fix this automatically. |
| No punctuation after number or at end of caption | NEEDS ATTENTION | The current captions end with periods. FoP states "nor is any punctuation to be placed at the end of the caption." |

### 6.3 Figure Placement and Accessibility

| Requirement | Status | Notes |
|---|---|---|
| Figures within body of text | PASS | Figures are included inline |
| Descriptive captions for accessibility | PASS | Captions describe figure content |
| Patterns in addition to colors | NEEDS ATTENTION | Cannot verify without viewing the PDFs, but the scripts should be checked to ensure figures use patterns/line styles in addition to colors for colorblind accessibility. |

**Action items:**
- [ ] Remove trailing periods from figure captions.
- [ ] Consider renaming figure files to `Fig1.pdf`, `Fig2.pdf`, etc.
- [ ] Verify that figures use distinguishable line styles (solid, dashed, dotted) in addition to color differentiation for accessibility.
- [ ] Consider converting figures to EPS format (optional).

---

## 7. Supplementary Information

| Requirement | Status | Notes |
|---|---|---|
| SI referenced in text | NEEDS ATTENTION | The manuscript references a "reproducibility script" and "accompanying repository" but does not use the FoP-prescribed "Online Resource" terminology. |

**Action items:**
- [ ] If the code repository is to serve as supplementary information, reference it as "Online Resource 1" in the text and provide a concise caption per FoP guidelines.
- [ ] Alternatively, if the repository is purely supplementary context (not formal SI), ensure the Data Availability statement covers this.

---

## 8. Scientific Style

| Requirement | Status | Notes |
|---|---|---|
| SI units | N/A | Dimensionless quantities throughout (information-theoretic) |
| Italic for variables, roman for operators | PASS | Standard mathematical notation used correctly throughout |
| Bold for vectors/tensors | PASS | `\bm{}` used appropriately |
| Standard function notation (cos, det, log, etc.) | PASS | `\log`, `\cos`, `\sin`, `\exp`, `\min`, `\max` used correctly |

---

## 9. Ethical Responsibilities

| Requirement | Status | Notes |
|---|---|---|
| Original work, not submitted elsewhere simultaneously | NEEDS ATTENTION | See Sec. 1 above re: PRA version |
| No plagiarism | PASS | Original research with proper citations |
| No salami-slicing | NEEDS ATTENTION | The paper references a companion "DLN compression" preprint by the same author. The two papers appear to address distinct questions (one cognitive/computational, one quantum-physical), so this should not constitute salami-slicing, but the author should be prepared to explain the distinction to editors. |
| Correct author group at submission | PASS | Single author |

---

## 10. Compliance with Ethical Standards

| Requirement | Status | Notes |
|---|---|---|
| Disclosure of potential conflicts | FAIL | Missing (see Sec. 2.6) |
| Research involving human participants | N/A | Theoretical physics |
| Informed consent | N/A | Theoretical physics |

---

## 11. Research Data Policy and Data Availability

| Requirement | Status | Notes |
|---|---|---|
| Data Availability Statement | FAIL | Not present in the manuscript |
| Data deposited in public repository | PASS | Code and data on GitHub with Zenodo DOI |
| Data citation with persistent identifier | PASS | DOI: 10.5281/zenodo.18610548 |

**Action items:**
- [ ] Add a formal Data Availability Statement to the manuscript (see Sec. 2.6 action items).

---

## 12. Open Access / Copyright Considerations

| Requirement | Status | Notes |
|---|---|---|
| License compatibility | NEEDS ATTENTION | The paper is currently CC BY 4.0 licensed. If submitting as Open Choice (OA), this is compatible. If submitting as a standard (non-OA) article, the author will need to grant Springer an exclusive license, which conflicts with the existing CC BY 4.0 license on the Zenodo deposit. |

**Action items:**
- [ ] Decide on Open Choice (OA) vs. standard submission. If OA, CC BY 4.0 is fine. If standard, be aware that the Zenodo preprint under CC BY 4.0 may create licensing questions.

---

## Priority Action-Item Checklist

### Must Fix (FAIL items -- submission will be returned without these)

1. [ ] **Add Keywords** (4--6 keywords after the abstract)
2. [ ] **Add Declarations section** before references:
   - Competing Interests statement
   - Funding statement
   - Data Availability statement
3. [ ] **Add Acknowledgments section** (even if brief)
4. [ ] **Trim abstract** to <= 250 words
5. [ ] **Confirm no simultaneous submission** to another journal

### Should Fix (NEEDS ATTENTION -- likely flagged by editors/reviewers)

6. [ ] **Shorten title** (currently very long; consider dropping the subtitle)
7. [ ] **Complete affiliation** with city, (state), country
8. [ ] **Define all abbreviations** at first use: CPTP, DAG, i.i.d., POVM, SBS (in abstract)
9. [ ] **Citation style**: verify numeric [1] format renders correctly; change `\citet` to `\citep` where needed for FoP's bracket-number style
10. [ ] **Reference formatting**: verify DOIs as full links, journal name abbreviations per ISSN LTWA
11. [ ] **Heading levels**: reduce to max 3 levels (remove `\paragraph` headings or promote them)
12. [ ] **Figure captions**: remove trailing periods; verify "Fig." format
13. [ ] **AI/LLM disclosure**: add statement if any AI tools were used
14. [ ] **Springer Nature LaTeX template**: consider adopting `sn-jnl.cls` for automatic compliance with many formatting requirements

### Nice to Have (will not block submission)

15. [ ] Rename figure files to `Fig1.pdf`, `Fig2.pdf`, etc.
16. [ ] Use "Online Resource" terminology for supplementary materials
17. [ ] Verify figure accessibility (patterns in addition to colors)
18. [ ] Resolve Open Choice vs. standard submission licensing

---

## Content-Level Observations (Beyond Formatting)

While the guidelines review above focuses on compliance, a few content observations relevant to a *Foundations of Physics* submission:

1. **Scope fit**: Foundations of Physics publishes work on "the conceptual bases and fundamental theories of modern physics." The paper's core contributions -- observer modeling in quantum Darwinism, SBS robustness, and Chernoff-optimal decoding -- fit this scope well. The DLN cognitive-framework connection (Secs. 3.3 and 7) is more interdisciplinary; FoP editors may or may not view it as central to the journal's mission. Consider whether to trim or expand the DLN discussion based on FoP's audience.

2. **The `\cite` vs `\citet` usage**: Throughout the paper, constructions like "synthesized by Zurek (2009)" use `\citet`. In FoP's numeric style, this becomes "synthesized by Zurek [3]" which is acceptable, but some instances like "the quantum Chernoff bound states that..." followed by a citation would need square brackets only.

3. **Self-citation balance**: The paper cites `WuDLNCompression2026preprint` (by the same author) 11 times. FoP guidelines explicitly warn against "excessive and inappropriate self-citation." While the citations appear substantively justified (the DLN framework is genuinely built upon), the author should be aware that reviewers may flag this.

4. **Preprint status of key reference**: `WuDLNCompression2026preprint` is cited as a bioRxiv preprint. FoP states the reference list "should only include works that are cited in the text and that have been published or accepted for publication." If this preprint has not been accepted, it may need to be cited as a personal communication or noted as "submitted" -- though many physics journals tolerate arXiv/bioRxiv preprint citations. Check FoP's specific practice.
