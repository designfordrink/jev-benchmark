# J02 HARTH — local dataset run

This runbook uses the user's local HARTH CSV directory directly. It does not depend on the Hugging Face Dataset Viewer.

## Expected local files

The current dataset contains these 22 subjects:

`S006,S008,S009,S010,S012,S013,S014,S015,S016,S017,S018,S019,S020,S021,S022,S023,S024,S025,S026,S027,S028,S029`.

The subject IDs are intentionally preserved as filenames; do not renumber them.

## Windows path

Default path used by the helper script:

`D:\Downloads\harth\harth`

Run from the repository root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-j02-harth-local.ps1
```

Or specify the directory explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-j02-harth-local.ps1 `
  -DatasetRoot "D:\Downloads\harth\harth"
```

## Stages

The script deliberately performs three stages:

1. `harth-validate`
   - required HARTH columns;
   - row counts;
   - timestamp parsing;
   - six accelerometer channels;
   - known labels;
   - non-finite feature values;
   - observed sampling rate.

2. `harth-inspect --subject S006`
   - reads actual windows from one real subject;
   - window size 128 samples;
   - stride 128;
   - reports the resulting window/label structure.

3. `harth-loso`
   - all 22 supplied subjects;
   - subject-disjoint LOSO;
   - Tiny MLP, 8 hidden units;
   - seed 0;
   - bounded smoke settings (500 train windows per training subject, 2000 test windows per held-out subject).

The smoke run is intentionally bounded. It is not yet the final benchmark result.

## Output

The script writes:

- `results/j02-harth/validation.json`
- `results/j02-harth/inspect-S006.json`
- `results/j02-harth/loso-tiny-mlp-seed0.json`

Do not publish the LOSO number as a final benchmark result until the dataset validation passes and the exact subject list, seed, windowing and sampling limits are recorded.

## Dataset provenance

The official UCI HARTH description specifies 22 subjects, two 3-axis accelerometers, separate CSV per subject, and a 50 Hz sampling rate. It defines the six accelerometer columns and the annotated activity labels used by J02.

Source: UCI HARTH, DOI 10.24432/C5NC90.
