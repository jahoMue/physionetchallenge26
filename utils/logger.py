"""
utils/logger.py
===============
Zentrales Logging-System für die gesamte Pipeline.
Loggt sowohl in die Konsole als auch in Dateien.
Erstellt pro Lauf und pro Patient separate Log-Einträge.
"""

import sys
from datetime import datetime
from loguru import logger as _logger

from config import LOG_DIR, LOG_LEVEL, LOG_TO_FILE, LOG_TO_CONSOLE


def setup_logger(module_name: str = "pipeline"):
    """
    Konfiguriert den Logger für ein bestimmtes Modul.
    
    Parameters
    ----------
    module_name : str
        Name des Moduls (wird im Dateinamen verwendet).
    
    Returns
    -------
    loguru.Logger
        Konfigurierter Logger.
    """
    # Entferne Standard-Handler
    _logger.remove()
    
    # Timestamp für Log-Datei
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Format
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[module]}</cyan> | "
        "<cyan>{extra[patient_id]}</cyan> | "
        "{message}"
    )
    
    # Konsolen-Handler
    if LOG_TO_CONSOLE:
        _logger.add(
            sys.stderr,
            format=log_format,
            level=LOG_LEVEL,
            colorize=True,
        )
    
    # Datei-Handler: Hauptlog
    if LOG_TO_FILE:
        _logger.add(
            LOG_DIR / f"{module_name}_{timestamp}.log",
            format=log_format,
            level=LOG_LEVEL,
            rotation="50 MB",
            retention="30 days",
            compression="zip",
            enqueue=True,
        )
        
        # Separater Error-Log
        _logger.add(
            LOG_DIR / f"{module_name}_{timestamp}_errors.log",
            format=log_format,
            level="WARNING",
            rotation="10 MB",
            retention="30 days",
            enqueue=True,
        )
    
    return _logger


def get_patient_logger(logger, module_name: str, patient_id: str):
    """
    Erstellt einen kontextualisierten Logger für einen bestimmten Patienten.
    
    Parameters
    ----------
    logger : loguru.Logger
        Basis-Logger.
    module_name : str
        Name des aktuellen Moduls.
    patient_id : str
        Patienten-ID.
    
    Returns
    -------
    loguru.Logger
        Logger mit Patient-Kontext.
    """
    return logger.bind(module=module_name, patient_id=patient_id)


class PipelineStats:
    """
    Sammelt Statistiken über die Pipeline-Verarbeitung pro Patient.
    """
    
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.stats = {
            "patient_id": patient_id,
            "total_segments": 0,
            "valid_ecg_segments": 0,
            "rejected_ecg_segments": 0,
            "valid_eeg_segments": 0,
            "rejected_eeg_segments": 0,
            "eeg_channels_used": [],
            "eeg_strategy": "",  # "averaged", "single", "none"
            "total_arousals": 0,
            "total_central_apneas": 0,
            "total_obstructive_apneas": 0,
            "total_hypopneas": 0,
            "total_sleep_duration_min": 0,
            "sleep_stages_found": [],
            "mean_ecg_sqi": 0.0,
            "mean_eeg_sqi": 0.0,
            "errors": [],
            "warnings": [],
        }
    
    def update(self, key: str, value):
        """Update a statistic."""
        if key in self.stats:
            if isinstance(self.stats[key], list):
                if isinstance(value, list):
                    self.stats[key].extend(value)
                else:
                    self.stats[key].append(value)
            else:
                self.stats[key] = value
    
    def increment(self, key: str, amount: int = 1):
        """Increment a numeric statistic."""
        if key in self.stats and isinstance(self.stats[key], (int, float)):
            self.stats[key] += amount
    
    def get_summary(self) -> dict:
        """Return summary statistics."""
        return self.stats.copy()
    
    def log_summary(self, logger):
        """Log the summary statistics."""
        logger.info("=" * 60)
        logger.info(f"Pipeline Summary for Patient: {self.patient_id}")
        logger.info("=" * 60)
        for key, value in self.stats.items():
            if key != "patient_id":
                logger.info(f"  {key}: {value}")
        logger.info("=" * 60)
