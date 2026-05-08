import pyro.poutine as poutine
from src.models.attacker_model import forward_model as attacker_model_fn
from src.models.goalkeeper_model import goalkeeper_model as goalkeeper_model_fn
from src.models.midfield_model import midfielder_model as midfielder_model_fn
from src.models.defender_model import defender_model as defender_model_fn
from src.models.combined_hierarchical_model import combined_hierarchical_model
from src.models.hierarchical_ard_model import hierarchical_model as hierarchical_model_fn

"""This file contains configurations for models, as we wrote them individually. 
It standardizes for evaluate_performance.py, such that it can easily be called from the command-line for all models."""

def combined_attacker_wrapper(**kwargs):
    hidden = [
        "mf_ohp", "mf_shots", "mf_fouled", "mf_xA", "mf_g", "mf_rating",
        "gk_saves", "gk_pass", "gk_br", "gk_gp", "gk_cs", "gk_rating",
        "def_touches", "def_duels", "def_passes", "def_clearances", "def_cs", "def_rating"
    ]
    return poutine.block(combined_hierarchical_model, hide=hidden)(attacker_data=kwargs)

def combined_midfielder_wrapper(**kwargs):
    hidden = [
        "att_dw", "att_br", "att_kp", "att_ts", "att_xa", "att_sot", "att_g", "att_rating",
        "gk_saves", "gk_pass", "gk_br", "gk_gp", "gk_cs", "gk_rating",
        "def_touches", "def_duels", "def_passes", "def_clearances", "def_cs", "def_rating"
    ]
    return poutine.block(combined_hierarchical_model, hide=hidden)(midfielder_data=kwargs)

def combined_goalkeeper_wrapper(**kwargs):
    hidden = [
        "att_dw", "att_br", "att_kp", "att_ts", "att_xa", "att_sot", "att_g", "att_rating",
        "mf_ohp", "mf_shots", "mf_fouled", "mf_xA", "mf_g", "mf_rating",
        "def_touches", "def_duels", "def_passes", "def_clearances", "def_cs", "def_rating"
    ]
    return poutine.block(combined_hierarchical_model, hide=hidden)(goalkeeper_data=kwargs)

def combined_defender_wrapper(**kwargs):
    hidden = [
        "att_dw", "att_br", "att_kp", "att_ts", "att_xa", "att_sot", "att_g", "att_rating",
        "mf_ohp", "mf_shots", "mf_fouled", "mf_xA", "mf_g", "mf_rating",
        "gk_saves", "gk_pass", "gk_br", "gk_gp", "gk_cs", "gk_rating"
    ]
    return poutine.block(combined_hierarchical_model, hide=hidden)(defender_data=kwargs)

def combined_all_wrapper(attacker_data=None, midfielder_data=None, goalkeeper_data=None, defender_data=None):
    return combined_hierarchical_model(
        attacker_data=attacker_data,
        midfielder_data=midfielder_data,
        goalkeeper_data=goalkeeper_data,
        defender_data=defender_data
    )

MODEL_CONFIGS = {
    "hierarchical": {
        "model_fn": hierarchical_model_fn,
        "is_hierarchical": True,
        "target": "rating",
    },
    "attacker": {
        "model_fn": attacker_model_fn,
        "features": ['groundDuelsWon', 'ballRecovery', 'keyPasses', 'expectedAssists', 'totalShots', 'shotsOnTarget', 'goals'],
        "pos_filter": 'F',
        "feature_map": {'groundDuelsWon': 'dw', 'ballRecovery': 'br', 'keyPasses': 'kp', 'expectedAssists': 'xa', 'totalShots': 'ts', 'shotsOnTarget': 'sot', 'goals': 'g_raw'},
        "target": "rating",
        "to_standardize": ['groundDuelsWon', 'ballRecovery', 'keyPasses', 'expectedAssists', 'totalShots', 'shotsOnTarget']
    },
    "midfielder": {
        "model_fn": midfielder_model_fn,
        "features": ["accurateOppositionHalfPasses", "shotsFromOutsideTheBox", "wasFouled", "expectedAssists", "goals"],
        "pos_filter": 'M',
        "feature_map": {'accurateOppositionHalfPasses': 'opp_half_passes', 'shotsFromOutsideTheBox': 'shots_outside', 'wasFouled': 'was_fouled', 'expectedAssists': 'xA', 'goals': 'goals'},
        "target": "rating",
        "to_standardize": [] 
    },
    "goalkeeper": {
        "model_fn": goalkeeper_model_fn,
        "features": ['saves', 'accuratePasses', 'ballRecovery', 'goalsPrevented', 'cleanSheet'],
        "pos_filter": 'G',
        "feature_map": {'saves': 'saves', 'accuratePasses': 'accuratePasses', 'ballRecovery': 'ballRecovery', 'goalsPrevented': 'goalsPrevented', 'cleanSheet': 'cleanSheet_raw'},
        "target": "rating",
        "to_standardize": ['saves', 'accuratePasses', 'ballRecovery', 'goalsPrevented']
    },
    "defender": {
        "model_fn": defender_model_fn,
        "features": ['touches', 'accuratePasses', 'totalDuelsWon', 'clearances', 'cleanSheet'],
        "pos_filter": 'D',
        "feature_map": {'touches': 'touches', 'accuratePasses': 'accuratePasses', 'totalDuelsWon': 'totalDuelsWon', 'clearances': 'clearances', 'cleanSheet': 'cleanSheet'},
        "target": "rating",
        "to_standardize": ['touches', 'accuratePasses', 'totalDuelsWon', 'clearances']
    },
    "combined_attacker": {
        "model_fn": combined_attacker_wrapper,
        "features": ['groundDuelsWon', 'ballRecovery', 'keyPasses', 'expectedAssists', 'totalShots', 'shotsOnTarget', 'goals'],
        "pos_filter": 'F',
        "feature_map": {'groundDuelsWon': 'dw', 'ballRecovery': 'br', 'keyPasses': 'kp', 'expectedAssists': 'xa', 'totalShots': 'ts', 'shotsOnTarget': 'sot', 'goals': 'g_raw'},
        "target": "rating",
        "to_standardize": ['groundDuelsWon', 'ballRecovery', 'keyPasses', 'expectedAssists', 'totalShots', 'shotsOnTarget']
    },
    "combined_midfielder": {
        "model_fn": combined_midfielder_wrapper,
        "features": ["accurateOppositionHalfPasses", "shotsFromOutsideTheBox", "wasFouled", "expectedAssists", "goals"],
        "pos_filter": 'M',
        "feature_map": {'accurateOppositionHalfPasses': 'opp_half_passes', 'shotsFromOutsideTheBox': 'shots_outside', 'wasFouled': 'was_fouled', 'expectedAssists': 'xA', 'goals': 'goals'},
        "target": "rating",
        "to_standardize": []
    },
    "combined_goalkeeper": {
        "model_fn": combined_goalkeeper_wrapper,
        "features": ['saves', 'accuratePasses', 'ballRecovery', 'goalsPrevented', 'cleanSheet'],
        "pos_filter": 'G',
        "feature_map": {'saves': 'saves', 'accuratePasses': 'accuratePasses', 'ballRecovery': 'ballRecovery', 'goalsPrevented': 'goalsPrevented', 'cleanSheet': 'cleanSheet_raw'},
        "target": "rating",
        "to_standardize": ['saves', 'accuratePasses', 'ballRecovery', 'goalsPrevented']
    },
    "combined_defender": {
        "model_fn": combined_defender_wrapper,
        "features": ['touches', 'accuratePasses', 'totalDuelsWon', 'clearances', 'cleanSheet'],
        "pos_filter": 'D',
        "feature_map": {'touches': 'touches', 'accuratePasses': 'accuratePasses', 'totalDuelsWon': 'totalDuelsWon', 'clearances': 'clearances', 'cleanSheet': 'cleanSheet'},
        "target": "rating",
        "to_standardize": ['touches', 'accuratePasses', 'totalDuelsWon', 'clearances']
    },
    "combined_all": {
        "model_fn": combined_all_wrapper,
        "is_combined_all": True,
        "positions": {
            "attacker": {
                "pos_filter": "F",
                "features": ['groundDuelsWon', 'ballRecovery', 'keyPasses', 'expectedAssists', 'totalShots', 'shotsOnTarget', 'goals'],
                "feature_map": {'groundDuelsWon': 'dw', 'ballRecovery': 'br', 'keyPasses': 'kp', 'expectedAssists': 'xa', 'totalShots': 'ts', 'shotsOnTarget': 'sot', 'goals': 'g_raw'},
                "to_standardize": ['groundDuelsWon', 'ballRecovery', 'keyPasses', 'expectedAssists', 'totalShots', 'shotsOnTarget'],
                "target_site": "att_rating",
                "target": "rating"
            },
            "midfielder": {
                "pos_filter": "M",
                "features": ["accurateOppositionHalfPasses", "shotsFromOutsideTheBox", "wasFouled", "expectedAssists", "goals"],
                "feature_map": {'accurateOppositionHalfPasses': 'opp_half_passes', 'shotsFromOutsideTheBox': 'shots_outside', 'wasFouled': 'was_fouled', 'expectedAssists': 'xA', 'goals': 'goals'},
                "to_standardize": [],
                "target_site": "mf_rating",
                "target": "rating"
            },
            "goalkeeper": {
                "pos_filter": "G",
                "features": ['saves', 'accuratePasses', 'ballRecovery', 'goalsPrevented', 'cleanSheet'],
                "feature_map": {'saves': 'saves', 'accuratePasses': 'accuratePasses', 'ballRecovery': 'ballRecovery', 'goalsPrevented': 'goalsPrevented', 'cleanSheet': 'cleanSheet_raw'},
                "to_standardize": ['saves', 'accuratePasses', 'ballRecovery', 'goalsPrevented'],
                "target_site": "gk_rating",
                "target": "rating"
            },
            "defender": {
                "pos_filter": "D",
                "features": ['touches', 'accuratePasses', 'totalDuelsWon', 'clearances', 'cleanSheet'],
                "feature_map": {'touches': 'touches', 'accuratePasses': 'accuratePasses', 'totalDuelsWon': 'totalDuelsWon', 'clearances': 'clearances', 'cleanSheet': 'cleanSheet'},
                "to_standardize": ['touches', 'accuratePasses', 'totalDuelsWon', 'clearances'],
                "target_site": "def_rating",
                "target": "rating"
            }
        },
        "target": "rating"
    }
}
