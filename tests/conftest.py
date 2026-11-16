import pytest

from nba import enrichment, pipeline, rules, scoring, synthetic, universe
from nba.config import ROOT, load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def rule_table(cfg):
    return rules.load_rules(ROOT / cfg["paths"]["rules"])


@pytest.fixture(scope="session")
def lines():
    return synthetic.make_lines(5_000, seed=3)


@pytest.fixture(scope="session")
def scored(lines, cfg, rule_table):
    return pipeline.build_scored(lines, cfg, rule_table)


@pytest.fixture(scope="session")
def flagged(lines, cfg):
    from nba import care
    uni = universe.build_universe(lines, 8)
    enr = enrichment.add_features(uni, uni["n_lines_person"])
    return scoring.build_flags(care.apply_care(enr, cfg), cfg)
