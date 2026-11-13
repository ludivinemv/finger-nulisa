
from pathlib import Path

VERSION = '2-1'
DEBUG = True

DATA_ORIGINAL_PATH = Path('../data/original')
DATA_PROCESSED_PATH = Path(f'../data/processed/version-{VERSION}')


LOG_PATH = Path(f"../data/outputs_python/logs/version-{VERSION}")

DEMO_PATH = Path(f'../data/outputs_python/demography/version-{VERSION}')
PRIMARY_PATH = Path(f'../data/outputs_python/primary_endpoints/version-{VERSION}')
SECONDARY_PATH = Path(f'../data/outputs_python/secondary_endpoints/version-{VERSION}')
EXPLORATORY_PATH = Path(f'../data/outputs_python/exploratory_endpoints/version-{VERSION}')
CROSS_PATH = Path(f'../data/outputs_python/cross_sectional_analyses/version-{VERSION}')
LONG_PATH = Path(f'../data/outputs_python/longitudinal_analyses/version-{VERSION}')

COLORINTER = "#CC79A7"
COLORCONTROL = "#4D4D4D"
COLORLOWADHERENCE = "#B0DDEB"


  