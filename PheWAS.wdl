version 1.0

## PheWAS via PheTK, scattered over datasets x PGS columns x sex strata.
## PGS columns are selected by regex against each cohort file's header.
## Requires PheTK >= 0.2.6 (fixes female-only phecodes leaking into male-only cohorts).

struct Dataset {
  String name            # e.g. "COMBINED" (used in output filenames)
  File   phecode_counts  # phecode_counts_<name>.tsv  (person_id, phecode, count)
  File   cohort          # <name>_cohort_phewas.tsv   (person_id, covariates, PGS columns)
}

## Reads the cohort header and returns the columns that fully match the regex,
## as a 2-column TSV: <column>\t<label>. Label is "" if not in pgs_labels.
task resolve_pgs {
  input {
    File   cohort
    String pgs_regex
    Array[String] exclude_cols            # covariates etc. never treated as PGS
    Map[String, String] pgs_labels
    String docker
  }

  File labels_file  = write_map(pgs_labels)
  File regex_file   = write_lines([pgs_regex])      # avoids shell-quoting the regex
  File exclude_file = write_lines(exclude_cols)

  command <<<
    set -euo pipefail
    cat > resolve.py << 'PYEOF'
import re, sys
import pandas as pd

cohort, regex_path, exclude_path, labels_path = sys.argv[1:5]
regex = open(regex_path).read().strip()
exclude = {l.strip() for l in open(exclude_path) if l.strip()}
cols = pd.read_csv(cohort, sep="\t", nrows=0).columns
pat = re.compile(regex)
hits = [c for c in cols if pat.fullmatch(c) and c not in exclude]
if not hits:
    raise SystemExit(f"No columns in {cohort} fully match regex '{regex}'")

labels = {}
with open(labels_path) as f:
    for line in f:
        line = line.rstrip("\n")
        if line:
            k, v = line.split("\t", 1)
            labels[k] = v

with open("pgs_table.tsv", "w") as out:
    for c in hits:
        out.write(f"{c}\t{labels.get(c, '')}\n")
print(f"{len(hits)} columns matched '{regex}': {hits}")
PYEOF
    python resolve.py "~{cohort}" "~{regex_file}" "~{exclude_file}" "~{labels_file}"
  >>>

  output {
    Array[Array[String]] pgs_table = read_tsv("pgs_table.tsv")
  }

  runtime {
    docker: docker
    cpu: 1
    memory: "4 GB"
    disks: "local-disk 20 HDD"
    preemptible: 2
  }
}

task run_phewas {
  input {
    String dataset_name
    File   phecode_counts
    File   cohort
    String pgs_id
    String pgs_label                    # may be ""
    String sex_stratum                  # "all", "F", or "M"

    Array[String] covariate_cols
    String id_col             = "person_id"
    String sex_col            = "sex_binary"
    Boolean male_as_one       = true
    String phecode_version    = "X"
    String method             = "logit"  # or "firth_logit"
    Int    min_cases          = 50
    Int    min_phecode_count  = 2
    String output_suffix      = ""      # e.g. "agesq"

    String docker
    Int    cpu                = 4
    Int    memory_gb          = 16
    Int    disk_gb            = 50
  }

  String out_name = dataset_name + "_" + sex_stratum + "_" + pgs_id
                    + (if pgs_label == "" then "" else "_" + pgs_label)
                    + (if output_suffix == "" then "" else "_" + output_suffix)
                    + "_phewas_results.tsv"

  command <<<
    set -euo pipefail

    cat > run_phewas.py << 'PYEOF'
import argparse
import pandas as pd
from phetk.phewas import PheWAS

ap = argparse.ArgumentParser()
ap.add_argument("--phecode-counts", required=True)
ap.add_argument("--cohort", required=True)
ap.add_argument("--pgs", required=True)
ap.add_argument("--covariates", required=True)        # comma-separated
ap.add_argument("--id-col", required=True)
ap.add_argument("--sex-col", required=True)
ap.add_argument("--male-as-one", required=True)
ap.add_argument("--stratum", required=True)           # all / F / M
ap.add_argument("--phecode-version", required=True)
ap.add_argument("--method", required=True)
ap.add_argument("--min-cases", type=int, required=True)
ap.add_argument("--min-phecode-count", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

male_as_one = a.male_as_one.lower() == "true"
covs = [c for c in a.covariates.split(",") if c]

df = pd.read_csv(a.cohort, sep="\t")

needed = [a.id_col, a.pgs, a.sex_col] + covs
missing = sorted(set(needed) - set(df.columns))
if missing:
    raise SystemExit(f"Cohort file is missing columns: {missing}")

# Drop rows with missing values in any model column, and say how many.
n0 = len(df)
df = df.dropna(subset=needed)
if len(df) < n0:
    print(f"WARNING: dropped {n0 - len(df)} of {n0} rows with missing values in model columns")

# Sex must be coded 0/1 for PheTK and for the stratum filter below.
sex_vals = set(df[a.sex_col].unique())
if not sex_vals <= {0, 1}:
    raise SystemExit(f"{a.sex_col} must be coded 0/1; found {sorted(sex_vals)}")
df[a.sex_col] = df[a.sex_col].astype(int)

if a.stratum in ("F", "M"):
    male_code = 1 if male_as_one else 0
    code = male_code if a.stratum == "M" else 1 - male_code
    df = df[df[a.sex_col] == code]
    covs = [c for c in covs if c != a.sex_col]   # constant within a stratum
elif a.stratum != "all":
    raise SystemExit(f"Unknown stratum: {a.stratum}")

if len(df) == 0:
    raise SystemExit(f"Stratum {a.stratum} has 0 participants - check {a.sex_col} coding / male_as_one")

# Covariates that are constant in this stratum would make the model singular.
const = [c for c in covs if df[c].nunique() <= 1]
if const:
    raise SystemExit(f"Covariates constant in stratum {a.stratum}: {const}")

cohort_path = "cohort_model.tsv"
df.to_csv(cohort_path, sep="\t", index=False)
print(f"Stratum={a.stratum}  N={len(df)}  covariates={len(covs)}  method={a.method}")

PheWAS(
    phecode_version=a.phecode_version,
    phecode_count_file_path=a.phecode_counts,
    cohort_file_path=cohort_path,
    covariate_cols=covs,
    independent_variable_of_interest=a.pgs,
    sex_at_birth_col=a.sex_col,
    male_as_one=male_as_one,
    min_cases=a.min_cases,
    min_phecode_count=a.min_phecode_count,
    method=a.method,
    output_file_path=a.out,
).run()
PYEOF

    pip show phetk | grep -i '^version' || true

    python run_phewas.py \
      --phecode-counts "~{phecode_counts}" \
      --cohort "~{cohort}" \
      --pgs "~{pgs_id}" \
      --covariates "~{sep=',' covariate_cols}" \
      --id-col "~{id_col}" \
      --sex-col "~{sex_col}" \
      --male-as-one "~{male_as_one}" \
      --stratum "~{sex_stratum}" \
      --phecode-version "~{phecode_version}" \
      --method "~{method}" \
      --min-cases ~{min_cases} \
      --min-phecode-count ~{min_phecode_count} \
      --out "~{out_name}"

    test -s "~{out_name}" || { echo "PheTK did not write ~{out_name}" >&2; ls -la >&2; exit 1; }
  >>>

  output {
    File result = out_name
  }

  runtime {
    docker: docker
    cpu: cpu
    memory: "~{memory_gb} GB"
    disks: "local-disk ~{disk_gb} HDD"
    preemptible: 2
  }
}

workflow PheWAS {
  input {
    Array[Dataset] datasets

    # Python-style regex, must match the WHOLE column name (re.fullmatch).
    String pgs_regex = "PGS.*"
    # Optional friendly names for output files, e.g. {"PGS002746": "ADHD"}
    Map[String, String] pgs_labels = {}

    Array[String] sex_strata = ["all"]          # any of "all", "F", "M"

    # Covariates: base list + PC1..PC<n_pcs>
    Array[String] base_covariates = ["age", "age_squared", "sex_binary", "visit_frequency", "record_depth"]
    Int           n_pcs           = 35
    String        pc_prefix       = "PC"

    String id_col            = "person_id"
    String sex_col           = "sex_binary"
    Boolean male_as_one      = true
    String phecode_version   = "X"
    String method            = "logit"
    Int    min_cases         = 50
    Int    min_phecode_count = 2
    String output_suffix     = ""

    # Pin a specific tag from hub.docker.com/r/phetk/phetk/tags for reproducibility.
    String docker = "phetk/phetk:latest"
    Int cpu       = 4
    Int memory_gb = 16
    Int disk_gb   = 50
  }

  scatter (i in range(n_pcs)) {
    String pc_name = "~{pc_prefix}~{i + 1}"
  }
  Array[String] covariate_cols = flatten([base_covariates, pc_name])

  scatter (ds in datasets) {
    call resolve_pgs {
      input:
        cohort       = ds.cohort,
        pgs_regex    = pgs_regex,
        exclude_cols = flatten([covariate_cols, [id_col, sex_col]]),
        pgs_labels   = pgs_labels,
        docker       = docker
    }

    scatter (row in resolve_pgs.pgs_table) {
      # row[0] = column name, row[1] = label ("" if none)
      String pgs_label = if length(row) > 1 then row[1] else ""

      scatter (stratum in sex_strata) {
        call run_phewas {
          input:
            dataset_name      = ds.name,
            phecode_counts    = ds.phecode_counts,
            cohort            = ds.cohort,
            pgs_id            = row[0],
            pgs_label         = pgs_label,
            sex_stratum       = stratum,
            covariate_cols    = covariate_cols,
            id_col            = id_col,
            sex_col           = sex_col,
            male_as_one       = male_as_one,
            phecode_version   = phecode_version,
            method            = method,
            min_cases         = min_cases,
            min_phecode_count = min_phecode_count,
            output_suffix     = output_suffix,
            docker            = docker,
            cpu               = cpu,
            memory_gb         = memory_gb,
            disk_gb           = disk_gb
        }
      }
    }
  }

  output {
    Array[File] results = flatten(flatten(run_phewas.result))
  }
}
