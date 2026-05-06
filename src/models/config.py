from src.models.attacker_model_basic import forward_model as attacker_model_fn
from src.models.goalkeeper_model import goalkeeper_model as goalkeeper_model_fn
from src.models.midfield_model import midfielder_model as midfielder_model_fn

MODEL_CONFIGS = {
    "attacker": {
        "model_fn": attacker_model_fn,
        "features": ['groundDuelsWon', 'ballRecovery', 'keyPasses', 'expectedAssists', 'totalShots', 'shotsOnTarget', 'goals'],
        "pos_filter": 'F',
        "feature_map": {
            'groundDuelsWon': 'dw',
            'ballRecovery': 'br',
            'keyPasses': 'kp',
            'expectedAssists': 'xa',
            'totalShots': 'ts',
            'shotsOnTarget': 'sot',
            'goals': 'g_raw'
        },
        "target": "rating",
        "to_standardize": ['groundDuelsWon', 'ballRecovery', 'keyPasses', 'expectedAssists', 'totalShots', 'shotsOnTarget']
    },
    "goalkeeper": {
        "model_fn": goalkeeper_model_fn,
        "features": ['saves', 'accuratePasses', 'ballRecovery', 'goalsPrevented', 'cleanSheet'],
        "pos_filter": 'G',
        "feature_map": {
            'saves': 'saves',
            'accuratePasses': 'accuratePasses',
            'ballRecovery': 'ballRecovery',
            'goalsPrevented': 'goalsPrevented',
            'cleanSheet': 'cleanSheet_raw'
        },
        "target": "rating",
        "to_standardize": ['saves', 'accuratePasses', 'ballRecovery', 'goalsPrevented']
    },
    "midfielder": {
        "model_fn": midfielder_model_fn,
        "features": ["accurateOppositionHalfPasses", "shotsFromOutsideTheBox", "wasFouled", "expectedAssists", "goals"],
        "pos_filter": 'M',
        "feature_map": {
            'accurateOppositionHalfPasses': 'opp_half_passes',
            'shotsFromOutsideTheBox': 'shots_outside',
            'wasFouled': 'was_fouled',
            'expectedAssists': 'xA',
            'goals': 'goals'
        },
        "target": "rating",
        "to_standardize": [] 
    }
}
