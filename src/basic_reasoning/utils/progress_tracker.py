"""Progress tracking for dataset collection."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional
import time

@dataclass
class DatasetProgress:
    """Track progress for a single dataset."""
    name: str
    expected_size: int
    current_size: int = 0
    start_time: Optional[datetime] = None
    last_update: Optional[datetime] = None
    
    @property
    def percentage(self) -> float:
        """Calculate completion percentage."""
        return (self.current_size / self.expected_size) * 100 if self.expected_size > 0 else 0
    
    @property
    def eta(self) -> Optional[timedelta]:
        """Calculate estimated time remaining."""
        if not self.start_time or not self.last_update or self.current_size == 0:
            return None
            
        elapsed = (self.last_update - self.start_time).total_seconds()
        if elapsed == 0:
            return None
            
        rate = self.current_size / elapsed
        remaining = (self.expected_size - self.current_size) / rate if rate > 0 else None
        return timedelta(seconds=remaining) if remaining is not None else None

class ProgressTracker:
    """Track progress across multiple datasets."""
    
    def __init__(self):
        """Initialize progress tracker."""
        self.datasets: Dict[str, DatasetProgress] = {}
        self.start_time = datetime.now()
    
    def register_dataset(self, name: str, expected_size: int) -> None:
        """
        Register a new dataset for tracking.
        
        Args:
            name: Dataset name
            expected_size: Expected number of documents
        """
        self.datasets[name] = DatasetProgress(
            name=name,
            expected_size=expected_size,
            start_time=datetime.now()
        )
    
    def update_progress(self, name: str, current_size: int) -> None:
        """
        Update progress for a dataset.
        
        Args:
            name: Dataset name
            current_size: Current number of documents
        """
        if name not in self.datasets:
            raise KeyError(f"Dataset {name} not registered")
            
        dataset = self.datasets[name]
        dataset.current_size = current_size
        dataset.last_update = datetime.now()
    
    def get_progress_report(self) -> str:
        """
        Generate a formatted progress report.
        
        Returns:
            Formatted progress report string
        """
        report = ["\nDataset Collection Progress:"]
        report.append("-" * 50)
        
        total_expected = sum(d.expected_size for d in self.datasets.values())
        total_current = sum(d.current_size for d in self.datasets.values())
        total_percentage = (total_current / total_expected) * 100 if total_expected > 0 else 0
        
        for dataset in self.datasets.values():
            eta_str = f"ETA: {dataset.eta}" if dataset.eta else "ETA: N/A"
            report.append(
                f"{dataset.name}: {dataset.current_size}/{dataset.expected_size} "
                f"({dataset.percentage:.1f}%) - {eta_str}"
            )
        
        report.append("-" * 50)
        report.append(
            f"Total Progress: {total_current}/{total_expected} "
            f"({total_percentage:.1f}%)"
        )
        
        return "\n".join(report)
    
    def print_progress(self) -> None:
        """Print current progress report."""
        print(self.get_progress_report()) 