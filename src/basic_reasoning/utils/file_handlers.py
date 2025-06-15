"""File handling utilities for dataset collection."""

import json
import csv
import gzip
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class FileHandler:
    """Handle file operations for dataset collection."""
    
    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """
        Ensure directory exists, create if it doesn't.
        
        Args:
            path: Directory path
            
        Returns:
            Path object for the directory
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
        """
        Save data as JSON file.
        
        Args:
            data: Data to save
            path: Output file path
            indent: JSON indentation level
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.info(f"Saved JSON to {path}")
    
    @staticmethod
    def load_json(path: Union[str, Path]) -> Any:
        """
        Load data from JSON file.
        
        Args:
            path: Input file path
            
        Returns:
            Loaded data
        """
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON from {path}")
        return data
    
    @staticmethod
    def save_jsonl(data: List[Dict], path: Union[str, Path]) -> None:
        """
        Save data as JSONL file.
        
        Args:
            data: List of dictionaries to save
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"Saved JSONL to {path}")
    
    @staticmethod
    def load_jsonl(path: Union[str, Path]) -> List[Dict]:
        """
        Load data from JSONL file.
        
        Args:
            path: Input file path
            
        Returns:
            List of loaded dictionaries
        """
        path = Path(path)
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        logger.info(f"Loaded JSONL from {path}")
        return data
    
    @staticmethod
    def save_csv(data: List[Dict], path: Union[str, Path], fieldnames: Optional[List[str]] = None) -> None:
        """
        Save data as CSV file.
        
        Args:
            data: List of dictionaries to save
            path: Output file path
            fieldnames: Optional list of field names
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if not fieldnames and data:
            fieldnames = list(data[0].keys())
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        logger.info(f"Saved CSV to {path}")
    
    @staticmethod
    def load_csv(path: Union[str, Path]) -> List[Dict]:
        """
        Load data from CSV file.
        
        Args:
            path: Input file path
            
        Returns:
            List of loaded dictionaries
        """
        path = Path(path)
        data = []
        with open(path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        logger.info(f"Loaded CSV from {path}")
        return data
    
    @staticmethod
    def compress_file(input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Compress a file using gzip.
        
        Args:
            input_path: Path to input file
            output_path: Optional path for compressed file
            
        Returns:
            Path to compressed file
        """
        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path.with_suffix(input_path.suffix + '.gz')
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(input_path, 'rb') as f_in:
            with gzip.open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        logger.info(f"Compressed {input_path} to {output_path}")
        return output_path
    
    @staticmethod
    def decompress_file(input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Decompress a gzipped file.
        
        Args:
            input_path: Path to compressed file
            output_path: Optional path for decompressed file
            
        Returns:
            Path to decompressed file
        """
        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path.with_suffix('')
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with gzip.open(input_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        logger.info(f"Decompressed {input_path} to {output_path}")
        return output_path 