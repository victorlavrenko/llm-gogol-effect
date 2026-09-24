# Paper sources

## ICLR 2027 submission

The OpenReview submission source is split across:

- `iclr_main.tex`
- `iclr_body1.tex`
- `iclr_body2.tex`
- `iclr_statements.tex`
- `iclr_refs.tex`
- `iclr_appendix.tex`
- `iclr2027_conference.sty`

Do **not** upload `paper/paper.pdf` to ICLR: it is the older non-anonymous manuscript.

The supported submission workflow is run from the repository root:

```bash
sh make_iclr_submission.sh
```

It builds `dist/iclr2027_submission.pdf` and `dist/iclr2027_supplement.zip`, scans both outputs for identifying strings, and enforces the OpenReview file-size limits. The CI workflow `.github/workflows/iclr2027-submission.yml` runs the frozen analysis reproduction before creating the same package.

## Earlier manuscript

`main.tex`, `figures.tex`, and `paper.pdf` are retained only as the earlier named/arXiv manuscript source. They are not the ICLR submission files.
