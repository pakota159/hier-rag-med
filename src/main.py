"""
Main entry point for HierRAGMed.
"""

import argparse
from pathlib import Path

from loguru import logger

from config import Config
from web import start_server


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="HierRAGMed - Medical RAG System")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the server on"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on"
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return

    config = Config(config_path)
    logger.info("Loaded configuration")

    # Start server
    logger.info(f"Starting server on {args.host}:{args.port}")
    start_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main() 