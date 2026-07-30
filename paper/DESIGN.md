# LaTeX white-paper design contract

- Format: approximately five US Letter pages, single column, including
  references.
- Source: `paper/main.tex` with section-level `\input{}` files.
- Voice: direct, empirical, and plain, following the supplied
  `final_project.tex`; define mechanics before interpreting them.
- Status: protocol until a complete external gateway run passes
  `validate`. Never treat the offline scripted smoke run as evidence.
- Typography: 10 pt article body, 0.72-inch margins, restrained navy/teal
  palette, compact tables, vector TikZ diagrams, and vector analysis charts.
- Required structure: standalone abstract; explicit research question; related
  work synthesized by concept; reproducible method; objective results gate;
  limitations; conclusion with no new claims.
- Tables: use only for architecture comparisons and empirical operating points.
- Figures: the study pipeline and shared-budget architecture diagram are
  protocol figures. Validated analysis produces PNG and PDF versions of accuracy,
  invariance, and accuracy-token plots.
- Citation rule: every external factual or methodological claim uses a verified
  primary or official source. The generated results section never creates a
  citation.
- Build: run `python paper/build_paper.py`, then compile `paper/main.tex` with
  `latexmk`. Render every page to PNG and inspect it before delivery.
