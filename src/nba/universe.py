"""Universe and eligibility.

Design decisions encoded here:
- Mark first, exclude later: every line is kept with the reason it is (in)eligible, so the funnel is auditable.
- Each exclusion list is joined by *its own* key (phone vs. person). Crossing everything by person
  excludes far too many lines.
- Candidacy for home offers is decided per *person*, not per line.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .quality import assert_same_rows


def person_profile(lines: pd.DataFrame) -> pd.DataFrame:
    """One row per person: how many lines, what they hold, and which home offer applies."""
    held = pd.crosstab(lines["person_id"], lines["line_type"]).reindex(columns=["POSTPAID", "PREPAID", "HOME"], fill_value=0)
    prof = pd.DataFrame({
        "n_lines_person": held.sum(axis=1),
        "has_postpaid": held["POSTPAID"] > 0,
        "has_prepaid": held["PREPAID"] > 0,
        "has_home": held["HOME"] > 0,
    })
    prof["home_status"] = np.select(
        [prof.has_postpaid & prof.has_home,
         prof.has_postpaid & ~prof.has_home,
         prof.has_home & prof.has_prepaid & ~prof.has_postpaid,
         prof.has_home & ~prof.has_prepaid & ~prof.has_postpaid],
        ["ALREADY_CONVERGENT", "MOBILE_WITHOUT_HOME", "HOME_WITH_PREPAID", "HOME_WITHOUT_MOBILE"],
        default="NO_HOME_OPPORTUNITY")
    return prof.reset_index()


def build_universe(lines: pd.DataFrame, mass_registration_min_lines: int = 8) -> pd.DataFrame:
    n0 = len(lines)
    prof = person_profile(lines)
    out = lines.merge(prof[["person_id", "n_lines_person", "home_status"]], on="person_id", how="left")
    assert_same_rows(n0, len(out), "universe: join with person profile")

    # A person-level list applies to every line of that person, wherever the flag happened to be recorded.
    out["suppressed_by_person"] = out.groupby("person_id")["suppressed_by_person"].transform("max")

    out["ineligible_reason"] = np.select(
        [~out["active"],
         out["suppressed_by_line"],
         out["suppressed_by_person"],
         out["n_lines_person"] >= mass_registration_min_lines],
        ["INACTIVE_LINE", "LINE_SUPPRESSION_LIST", "PERSON_SUPPRESSION_LIST", "MASS_REGISTRATION"],
        default="")
    out["is_eligible"] = out["ineligible_reason"].eq("")
    out["ineligible_reason"] = out["ineligible_reason"].replace("", None)
    return out
