# SJTU-AI1201-Resources

This repository contains final review materials for the SJTU AI1201 course.

The materials are prepared for an introductory artificial intelligence course and are mainly intended to help students review before the final exam. The current focus is on concept-oriented review notes, simple formula checks, and selected figures used in the review handout.

## Contents

```text
output/
  期末复习课详细提纲.md        # Main detailed review outline
  期末复习课详细提纲.pdf       # Exported PDF version for distribution
  期末考试复习要点.md          # Earlier/high-level review notes
  复习提纲整理工作流程.md      # Workflow for organizing later chapters
  assets/                     # Figures extracted or prepared for the handout
```

## Review Scope

The review materials are organized around the final exam scope:

- Residual network basics
- Convolutional neural networks
- Recurrent neural networks and LSTM
- Transformer
- Neural network training phenomena
- Parameter condensation and flatness of solutions
- Large model basics
- Reinforcement learning

The notes are written for lower-division undergraduate students, so the emphasis is on:

- Definitions and core concepts
- Conceptual distinctions
- Simple formula recognition
- Basic CNN size and parameter calculations
- Short thinking questions for classroom review

## PDF Export

The main handout is maintained in Markdown:

```text
output/期末复习课详细提纲.md
```

The exported PDF is:

```text
output/期末复习课详细提纲.pdf
```

In the local working directory, the export script is:

```bash
scripts/export_md_pdf.sh
```

The script requires `pandoc` and `xelatex`. The script itself is not tracked in this repository because this repo is intended to store the review materials rather than the local build environment.

## Notes

- Original course slide PDFs are not tracked in this repository.
- Extracted figures used in the handout are tracked under `output/assets/`.
- This repository is intended for course review material management and iterative editing.
