# Changes in thesis_v2

`thesis_v2/` is a complete, compilable copy of the thesis. The original `thesis/` folder is
untouched. Build with `pdflatex main`, `bibtex main`, then `pdflatex main` twice. It compiles to
117 pages with no errors, no undefined references and no table wider than the page.

## 1. New structure

- **Split files.** Chapters 1 and 2 moved out of `main.tex` into `chapter1.tex` and
  `chapter2.tex`. `main.tex` now holds only the title page, the front matter, the abstract and
  the `\input` lines.
- **Chapter 2 order.** It now runs from the general to the specific:
  1. Readability assessment in general: formulas, machine learning, deep learning, transformers.
  2. A new subsection on the ordinal and ranking losses used for readability in other languages.
  3. Arabic readability: what makes it different from English.
  4. **Arabic readability projects before BAREC (new):**
     - Arabic formulas: Arability, AARI and OSMAN.
     - Feature-based classifiers: Saddiki et al. and Khallaf and Sharoff.
     - Lexicons and corpora: the SAMER lexicon and corpus, Liberato et al., DARES and ReadMe++.
     - A summary table of these projects.
     - A list of the six gaps they leave.
  5. The BAREC corpus, expanded with the guidelines, annotators, sources, agreement and a table
     showing how BAREC closes each gap.
  6. **Input representation and encoder (new):** why D3Tok and why AraBERTv2, with the BAREC
     paper's four-encoder by four-input QWK table as evidence.
  7. Ordinal regression methods, now including CORN and pairwise ranking.
  8. The BAREC 2025 shared-task systems (unchanged table).
  9. **Positioning of this thesis (new).**

The limitations of earlier projects are stated as limitations for fine-grained sentence-level
modelling, not as mistakes. Most of those projects were built for a different purpose.

## 2. New-loss results added (CORN, RANK, PLR)

- **Result files.** Your three JSON files are stored in `final/gpu code results/new losses/`.
- **Generator.** `gen_newloss_numbers.py` reads them, plus the v3 MSE and CORAL controls, and
  writes `newlossnumbers.tex`, `tablenewlossdev.tex`, `tablenewlosstest.tex` and
  `tablenewlossdelta.tex`. Every new number in the text comes from these files. Run
  `python3 gen_newloss_numbers.py` again if a result file changes.
- **Chapter 3.** New section, "Loss Functions Added from Readability Research in Other
  Languages". It covers CORN's head, loss and decoding, the RANK and PLR pairwise losses, their
  controls, the true batch of 64, and the two adaptations that are our own choices.
- **Chapter 4.** New section with the three tables and the discussion:
  - RANK joins the leading group on test QWK.
  - PLR has the fewest errors of five levels or more of any run, on both splits.
  - CORN trades 1.76 test QWK points for 20 points of exact accuracy and the lowest mean
    distance. This shows that CORAL's shared projection is what makes it behave like regression.
  - A new limitation item covers these runs.
- **Chapters 1 and 5 and the abstract.** Updated research questions, objectives, contributions,
  findings and future work, plus the run table and implementation notes in chapter 3.

## 3. Errors fixed

- **Compile error.** The abstract had `\cutMoveFineCoral\under`, where `\under` is an undefined
  command. It is fixed.
- **Leftover notes.** Removed the stray line "One thing to check: this section says four
  ablations, but the abstract says three" and a duplicated contribution bullet. The ablation
  count is now consistent: three in the fine-tuning track and one in the frozen track.
- **Duplicates.** Removed a duplicated paragraph in the AraBERTv2 subsection.
- **Typos.** Fixed "The The", "ssummarizes", Markdown asterisks in "(meaning *and*)", a missing
  full stop, and missing spaces before and after several citations.
- **SAMER lexicon size.** Changed from "40,687 lemma and POS pairs" to "about 26,000 lemmas", as
  stated in the LREC 2020 paper. If you have a source for 40,687, for example a later release,
  you can put it back with that citation.
- **Deutsch et al. 2020.** They found that linguistic features add little to transformers. The
  text now says so.
- **OSMAN.** The old chapter 2 summary mentioned OSMAN "among the formulas" without describing
  it. It is now described.
- **Transliteration.** Corrected "walfidda" to *walfiḍḍa*.

## 4. Layout

- **Page width.** All tables now fit the text width. This covers the main results, selection,
  CPU and new-loss tables through `adjustbox`, plus narrower columns in the chapter 2 and 3
  tables. The main, selection and CPU tables were already too wide in the original.
- **Acronyms.** Added CORN, CEFR, SAMER, DARES, OSMAN, AARI, RANK and PLR.
- **Title page.** Removed the commented-out title page left over from an earlier template
  ("Strategies for Arabic Readability Modeling").

## 5. Bibliography

Added seven entries: `tahathomure2017`, `lim-lee-2024-improving`, `trokhymovych2024readability`,
`shi2023corn`, `li-etal-2022-unified`, `zeng-etal-2022-enhancing` and `burges2005learning`.

## 6. Later fixes

- **Run count.** Chapter 4 now opens with "the fourteen training runs", matching chapter 3.
- **CORN check.** Chapter 3 now says that the CORN loss and its decoding are checked against the
  authors' published training loop and the chain rule, as Part D does before training.
- **Reported losses.** Only CORN, RANK and PLR are reported. TSOFT12, LBL10 and WKL are not part
  of the thesis.

## 7. Things to check

- **Ranking margin.** The RANK margin of 0.5 comes from the research summary of Trokhymovych et
  al. Confirm it in their paper.
- **DARES labels.** The statement that each DARES text is labelled with the grade of its
  textbook follows the dataset description. Check it in the DARES paper if you want to quote it.
- **Dedication pages.** The two empty dedication pages are left as they were, one per student.
