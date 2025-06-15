"""Logging configuration for the basic reasoning module."""

import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logging(log_dir: str = "logs") -> None:
    """
    Set up logging configuration for all modules.
    
    Args:
        log_dir: Directory to store log files
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create handlers
    handlers = {
        'data_collection': logging.FileHandler(
            log_path / f'data_collection_{datetime.now().strftime("%Y%m%d")}.log'
        ),
        'medreason': logging.FileHandler(
            log_path / f'medreason_{datetime.now().strftime("%Y%m%d")}.log'
        ),
        'msdiagnosis': logging.FileHandler(
            log_path / f'msdiagnosis_{datetime.now().strftime("%Y%m%d")}.log'
        ),
        'pmc_patients': logging.FileHandler(
            log_path / f'pmc_patients_{datetime.now().strftime("%Y%m%d")}.log'
        ),
        'drugbank': logging.FileHandler(
            log_path / f'drugbank_{datetime.now().strftime("%Y%m%d")}.log'
        ),
        'validation': logging.FileHandler(
            log_path / f'validation_{datetime.now().strftime("%Y%m%d")}.log'
        )
    }
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    
    # Configure loggers
    loggers = {
        'data_collection': logging.getLogger('data_collection'),
        'medreason': logging.getLogger('medreason'),
        'msdiagnosis': logging.getLogger('msdiagnosis'),
        'pmc_patients': logging.getLogger('pmc_patients'),
        'drugbank': logging.getLogger('drugbank'),
        'validation': logging.getLogger('validation')
    }
    
    # Set up each logger
    for name, logger in loggers.items():
        logger.setLevel(logging.INFO)
        
        # Add file handler
        handlers[name].setFormatter(file_formatter)
        logger.addHandler(handlers[name])
        
        # Add console handler
        logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False 