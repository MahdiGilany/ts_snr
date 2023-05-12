from collections import namedtuple
from src.data.database import connect
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.data.exact.resources import metadata


Splits = namedtuple("Splits", ["train", "val", "test"])


# def get_patient_splits(fold, n_folds=5):
#     """Returns the list of patient ids for the train, val, and test splits."""
#
#     with connect() as conn:
#         table = pd.read_sql("SELECT id, center FROM Patient", conn)
#
#     kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
#     for i, (train_idx, test_idx) in enumerate(kfold.split(table, table["center"])):
#         if i == fold:
#             train, test = table.iloc[train_idx], table.iloc[test_idx]
#             break
#
#     train, val = train_test_split(
#         train, test_size=0.2, random_state=0, stratify=train["center"]
#     )
#     splits = Splits(list(train["id"]), list(val["id"]), list(test["id"]))
#     # make sure there is no overlap between splits
#     assert len(set(splits.train) & set(splits.val)) == 0
#     assert len(set(splits.train) & set(splits.test)) == 0
#     assert len(set(splits.val) & set(splits.test)) == 0
#
#     return splits


def get_patient_splits(fold, n_folds=5):
    """returns the list of patient specifiers for the train, val, and test splits."""

    table = metadata()
    table = table.groupby("patient_specifier").first().reset_index()

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    for i, (train_idx, test_idx) in enumerate(kfold.split(table, table["center"])):
        if i == fold:
            train, test = table.iloc[train_idx], table.iloc[test_idx]
            break

    train, val = train_test_split(
        train, test_size=0.2, random_state=0, stratify=train["center"]
    )
    splits = Splits(
        list(train["patient_specifier"]),
        list(val["patient_specifier"]),
        list(test["patient_specifier"]),
    )
    assert len(set(splits.train) & set(splits.val)) == 0
    assert len(set(splits.train) & set(splits.test)) == 0
    assert len(set(splits.val) & set(splits.test)) == 0

    return splits


def get_cores_for_patients(patients):
    """Returns the list of core specifiers for the given list of patient specifiers."""
    table = metadata()
    table = table.query("patient_specifier in @patients")
    return table.core_specifier.to_list()


def _select_malignant_or_pure_benign_cores():
    """Selects malignant cores or cores from patients with no malignant cores.
    In other words, this function excludes benign cores from patients with at least one malignant core."""

    QUERY = """SELECT c.id as core_id
FROM Core c
    LEFT JOIN (
        SELECT p.id as patient_id,
            SUM(c.grade <> "Benign") as num_malignant
        FROM Core c
            LEFT JOIN Patient p ON c.patient_id = p.id
        GROUP BY p.id
    ) AS T ON c.patient_id = T.patient_id
WHERE c.grade <> "BENIGN"
    OR T.num_malignant <=> 0"""

    with connect() as conn:
        table = pd.read_sql(QUERY, conn)
    return table


def remove_benign_cores_from_positive_patients(cores):
    """Returns the list of cores in the given list that are either malignant or from patients with no malignant cores."""
    table = metadata().copy()
    table["positive"] = table.grade.apply(lambda g: 0 if g == "Benign" else 1)
    num_positive_for_patient = table.groupby("patient_specifier").positive.sum()
    num_positive_for_patient.name = "patients_positive"
    table = table.join(num_positive_for_patient, on="patient_specifier")
    ALLOWED = table.query(
        "positive == 1 or patients_positive == 0"
    ).core_specifier.to_list()

    return [core for core in cores if core in ALLOWED]


def remove_cores_below_threshold_involvement(cores, threshold_pct):
    """Returns the list of cores with at least the given percentage of cancer cells."""
    table = metadata()
    ALLOWED = table.query(
        "grade == 'Benign' or pct_cancer >= @threshold_pct"
    ).core_specifier.to_list()
    return [core for core in cores if core in ALLOWED]


def undersample_benign(cores, seed=0, benign_to_cancer_ratio=1):
    """Returns the list of cores with the same cancer cores and the benign cores undersampled to the given ratio."""

    table = metadata()
    benign = table.query('grade == "Benign"').core_specifier.to_list()
    cancer = table.query('grade != "Benign"').core_specifier.to_list()
    import random

    cores_benign = [core for core in cores if core in benign]
    cores_cancer = [core for core in cores if core in cancer]
    rng = random.Random(seed)
    cores_benign = rng.sample(
        cores_benign, int(len(cores_cancer) * benign_to_cancer_ratio)
    )

    return [core for core in cores if core in cores_benign or core in cores_cancer]


def undersample_benign_as_kfold(cores, seed=0, benign_to_cancer_ratio=1):
    """
    Returns a list of lists where each list is a fold of the same cancer cores and the benign cores undersampled to the given ratio.

    """
    table = metadata()
    benign = table.query('grade == "Benign"').core_specifier.to_list()
    cancer = table.query('grade != "Benign"').core_specifier.to_list()
    import random

    cores_benign = [core for core in cores if core in benign]
    cores_cancer = [core for core in cores if core in cancer]

    num_benign_cores_per_fold = int(len(cores_cancer) * benign_to_cancer_ratio)
    num_folds = int(len(cores_benign) / num_benign_cores_per_fold)

    from sklearn.model_selection import KFold

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits = []
    for train_idx, test_idx in kf.split(cores_benign):
        benign_for_fold = [cores_benign[i] for i in test_idx]
        splits.append(
            [core for core in cores if core in benign_for_fold or core in cores_cancer]
        )

    return splits


def _get_core_specifiers_for_core_ids():
    """Returns a table with the core specifier for each core id."""

    QUERY = """SELECT c.id as id,
    CONCAT(
        p.center,
        '-',
        LPAD(p.patient_number, 4, '0'),
        '_',
        c.biopsy_location
    ) as core_specifier
FROM Patient p
    LEFT JOIN Core c ON c.patient_id = p.id
ORDER BY core_specifier;"""

    with connect() as conn:
        return pd.read_sql(QUERY, conn, index_col="id")


def select_cores(
    fold,
    minimum_involvement=40,
    remove_benign_from_positive_patients=True,
    undersample_benign=False,
    undersample_benign_seed=0,
):
    train, val, test = get_patient_splits(fold)

    with connect() as conn:
        table = pd.read_sql(
            "SELECT Core.id as id, Patient.id as patient_id, involvement, grade FROM Core LEFT JOIN Patient ON Core.patient_id = Patient.id",
            conn,
            index_col="id",
        )

    table["split"] = table["patient_id"].apply(
        lambda x: "train"
        if x in train
        else "val"
        if x in val
        else "test"
        if x in test
        else "unknown"
    )

    if minimum_involvement:
        table = table.query("involvement == 0 or involvement >= @minimum_involvement")
    if remove_benign_from_positive_patients:
        cores_to_keep = _select_malignant_or_pure_benign_cores()["core_id"]
        table = table.query("id in @cores_to_keep")

    table_cancer = table.query("grade != 'Benign'")
    table_benign = table.query("grade == 'Benign'")
    all_table = []
    for split in ["train", "val", "test"]:
        table_cancer_split = table_cancer.query("split == @split")
        table_benign_split = table_benign.query("split == @split")
        if undersample_benign:
            table_benign_split = table_benign_split.sample(
                len(table_cancer_split), random_state=undersample_benign_seed
            )
        table = pd.concat([table_cancer_split, table_benign_split])
        all_table.append(table)

    table = pd.concat(all_table)
    table = table.sort_values(by="id")
    # table.set_index("id", inplace=True)
    table = table[["split"]]

    core_specifiers = _get_core_specifiers_for_core_ids()
    table = table.join(core_specifiers)

    return table
