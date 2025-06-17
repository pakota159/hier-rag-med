#!/usr/bin/env python3
"""
Model comparison script for HierRAGMed evaluation system.
Provides detailed comparative analysis between KG and Hierarchical systems.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import json
import pandas as pd
from datetime import datetime
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import evaluation components
from src.evaluation.evaluators.comparative_evaluator import ComparativeEvaluator
from src.evaluation.utils.statistical_analysis import StatisticalAnalysis
from src.evaluation.utils.visualization import EvaluationVisualizer
from src.evaluation.utils.report_generator import ReportGenerator


def load_evaluation_results(results_dir: Path) -> Dict:
    """Load existing evaluation results from directory."""
    try:
        # Look for the most recent results file
        result_files = list(results_dir.glob("evaluation_results_*.json"))
        if not result_files:
            # Try default filename
            default_file = results_dir / "evaluation_results.json"
            if default_file.exists():
                result_files = [default_file]
        
        if not result_files:
            raise FileNotFoundError("No evaluation results found")
        
        # Use most recent file
        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            results = json.load(f)
        
        logger.info(f"✅ Loaded evaluation results from {latest_file}")
        return results
        
    except Exception as e:
        logger.error(f"❌ Failed to load evaluation results: {e}")
        raise


def compare_models(
    results_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    significance_level: float = 0.05
) -> Dict:
    """
    Compare performance between KG and Hierarchical systems.
    
    Args:
        results_path: Path to evaluation results JSON file
        config_path: Path to evaluation config file
        output_dir: Output directory for comparison results
        significance_level: Statistical significance threshold
        
    Returns:
        Dict containing detailed comparison analysis
    """
    
    # Load configuration
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load evaluation results
    if results_path is None:
        results_dir = Path(config["results_dir"])
        evaluation_results = load_evaluation_results(results_dir)
    else:
        with open(results_path, 'r') as f:
            evaluation_results = json.load(f)
    
    # Set output directory
    if output_dir is None:
        output_dir = Path(config["results_dir"]) / "comparative"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🔄 Starting comprehensive model comparison...")
    
    # Initialize comparative evaluator
    comparative_evaluator = ComparativeEvaluator(config)
    
    # Perform detailed comparison
    comparison_results = comparative_evaluator.compare_systems(
        evaluation_results,
        significance_level=significance_level
    )
    
    # Statistical analysis
    logger.info("📊 Performing statistical analysis...")
    stats_analyzer = StatisticalAnalysis(config)
    statistical_results = stats_analyzer.analyze_results(evaluation_results)
    
    # Generate visualizations
    logger.info("📈 Generating comparison visualizations...")
    visualizer = EvaluationVisualizer(config)
    visualizations = visualizer.create_comparison_plots(
        evaluation_results,
        output_dir=output_dir
    )
    
    # Combine all analysis
    comprehensive_comparison = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "comparison_type": "kg_vs_hierarchical",
            "significance_level": significance_level,
            "source_results": str(results_path) if results_path else "latest"
        },
        "performance_comparison": comparison_results,
        "statistical_analysis": statistical_results,
        "visualizations": visualizations,
        "summary": _generate_comparison_summary(comparison_results, statistical_results)
    }
    
    # Save comparison results
    comparison_file = output_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(comparison_file, 'w') as f:
        json.dump(comprehensive_comparison, f, indent=2)
    
    # Generate comparison report
    logger.info("📝 Generating comparison report...")
    report_generator = ReportGenerator(config)
    report_generator.generate_comparison_report(
        comprehensive_comparison,
        output_dir=output_dir
    )
    
    logger.info(f"✅ Comparison completed! Results saved to {output_dir}")
    return comprehensive_comparison


def _generate_comparison_summary(comparison_results: Dict, statistical_results: Dict) -> Dict:
    """Generate executive summary of model comparison."""
    
    summary = {
        "overall_winner": None,
        "benchmark_winners": {},
        "significant_differences": [],
        "key_findings": [],
        "recommendations": []
    }
    
    # Determine overall winner
    kg_wins = 0
    hierarchical_wins = 0
    
    for benchmark, results in comparison_results.get("benchmark_comparison", {}).items():
        if "winner" in results:
            if results["winner"] == "kg_system":
                kg_wins += 1
                summary["benchmark_winners"][benchmark] = "KG System"
            elif results["winner"] == "hierarchical_system":
                hierarchical_wins += 1
                summary["benchmark_winners"][benchmark] = "Hierarchical System"
            else:
                summary["benchmark_winners"][benchmark] = "Tie"
    
    if hierarchical_wins > kg_wins:
        summary["overall_winner"] = "Hierarchical System"
    elif kg_wins > hierarchical_wins:
        summary["overall_winner"] = "KG System" 
    else:
        summary["overall_winner"] = "Tie"
    
    # Identify significant differences
    for benchmark, stats in statistical_results.get("significance_tests", {}).items():
        if stats.get("p_value", 1.0) < 0.05:
            summary["significant_differences"].append({
                "benchmark": benchmark,
                "p_value": stats["p_value"],
                "effect_size": stats.get("effect_size", "unknown")
            })
    
    # Generate key findings
    if summary["overall_winner"] == "Hierarchical System":
        summary["key_findings"].append(
            "Hierarchical 3-tier reasoning system outperforms KG system overall"
        )
    
    if "medreason" in summary["benchmark_winners"]:
        if summary["benchmark_winners"]["medreason"] == "Hierarchical System":
            summary["key_findings"].append(
                "Hierarchical system excels at clinical reasoning tasks"
            )
    
    if "pubmedqa" in summary["benchmark_winners"]:
        winner = summary["benchmark_winners"]["pubmedqa"]
        summary["key_findings"].append(
            f"{winner} performs better on research literature tasks"
        )
    
    # Generate recommendations
    if summary["overall_winner"] == "Hierarchical System":
        summary["recommendations"].append(
            "Deploy hierarchical system for production medical QA"
        )
        summary["recommendations"].append(
            "Further optimize 3-tier reasoning architecture"
        )
    else:
        summary["recommendations"].append(
            "Investigate hybrid approach combining both systems"
        )
    
    return summary


def generate_comparison_report(
    comparison_results: Dict,
    output_path: Path,
    format: str = "html"
) -> None:
    """Generate human-readable comparison report."""
    
    if format == "html":
        _generate_html_report(comparison_results, output_path)
    elif format == "markdown":
        _generate_markdown_report(comparison_results, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _generate_html_report(comparison_results: Dict, output_path: Path) -> None:
    """Generate HTML comparison report."""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>HierRAGMed Model Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
            .summary {{ background: #ecf0f1; padding: 15px; margin: 20px 0; border-radius: 5px; }}
            .benchmark {{ margin: 20px 0; padding: 15px; border: 1px solid #bdc3c7; border-radius: 5px; }}
            .winner {{ color: #27ae60; font-weight: bold; }}
            .metric {{ margin: 10px 0; }}
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧠 HierRAGMed Model Comparison Report</h1>
            <p>Generated: {comparison_results['metadata']['timestamp']}</p>
        </div>
        
        <div class="summary">
            <h2>📊 Executive Summary</h2>
            <p><strong>Overall Winner:</strong> <span class="winner">{comparison_results['summary']['overall_winner']}</span></p>
            
            <h3>Benchmark Winners:</h3>
            <ul>
    """
    
    for benchmark, winner in comparison_results['summary']['benchmark_winners'].items():
        html_content += f"<li><strong>{benchmark.upper()}:</strong> {winner}</li>"
    
    html_content += """
            </ul>
            
            <h3>Key Findings:</h3>
            <ul>
    """
    
    for finding in comparison_results['summary']['key_findings']:
        html_content += f"<li>{finding}</li>"
    
    html_content += """
            </ul>
        </div>
        
        <!-- Add detailed benchmark comparison tables here -->
        
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)


def _generate_markdown_report(comparison_results: Dict, output_path: Path) -> None:
    """Generate Markdown comparison report."""
    
    markdown_content = f"""# 🧠 HierRAGMed Model Comparison Report

**Generated:** {comparison_results['metadata']['timestamp']}

## 📊 Executive Summary

**Overall Winner:** {comparison_results['summary']['overall_winner']}

### Benchmark Winners

"""
    
    for benchmark, winner in comparison_results['summary']['benchmark_winners'].items():
        markdown_content += f"- **{benchmark.upper()}:** {winner}\n"
    
    markdown_content += "\n### Key Findings\n\n"
    
    for finding in comparison_results['summary']['key_findings']:
        markdown_content += f"- {finding}\n"
    
    markdown_content += "\n### Recommendations\n\n"
    
    for rec in comparison_results['summary']['recommendations']:
        markdown_content += f"- {rec}\n"
    
    markdown_content += "\n## 📈 Detailed Performance Comparison\n\n"
    
    # Add benchmark-specific details
    for benchmark, results in comparison_results.get('performance_comparison', {}).get('benchmark_comparison', {}).items():
        markdown_content += f"### {benchmark.upper()} Results\n\n"
        
        if 'scores' in results:
            markdown_content += "| Model | Score | Performance |\n"
            markdown_content += "|-------|-------|-------------|\n"
            for model, score in results['scores'].items():
                performance = "🔥 Excellent" if score > 75 else "✅ Good" if score > 65 else "⚠️ Needs Improvement"
                markdown_content += f"| {model} | {score:.2f}% | {performance} |\n"
        
        markdown_content += "\n"
    
    markdown_content += """
## 🎯 SOTA Comparison

| Benchmark | Our Best | SOTA | Gap |
|-----------|----------|------|-----|
| MIRAGE | TBD | 74.8% | TBD |
| MedReason | TBD | 71.3% | TBD |
| PubMedQA | TBD | 78.2% | TBD |
| MS MARCO | TBD | 0.35 | TBD |

## 📋 Next Steps

1. **Analyze failure cases** in lower-performing benchmarks
2. **Optimize retrieval** for better MS MARCO performance  
3. **Enhance reasoning** for clinical task improvement
4. **Conduct error analysis** on specific question types
5. **Implement improvements** based on comparative insights
"""
    
    with open(output_path, 'w') as f:
        f.write(markdown_content)


def main():
    """Command line interface for model comparison."""
    parser = argparse.ArgumentParser(description="Compare HierRAGMed models")
    
    parser.add_argument(
        "--results",
        type=Path,
        help="Path to evaluation results JSON file"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to evaluation config file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for comparison results"
    )
    
    parser.add_argument(
        "--significance-level",
        type=float,
        default=0.05,
        help="Statistical significance threshold (default: 0.05)"
    )
    
    parser.add_argument(
        "--format",
        choices=["html", "markdown", "both"],
        default="both",
        help="Report format to generate"
    )
    
    args = parser.parse_args()
    
    try:
        # Run comparison
        comparison_results = compare_models(
            results_path=args.results,
            config_path=args.config,
            output_dir=args.output_dir,
            significance_level=args.significance_level
        )
        
        # Generate additional reports if specified
        if args.output_dir:
            output_dir = args.output_dir
        else:
            config_path = args.config or Path(__file__).parent / "config.yaml"
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            output_dir = Path(config["results_dir"]) / "comparative"
        
        if args.format in ["html", "both"]:
            html_path = output_dir / "comparison_report.html"
            generate_comparison_report(comparison_results, html_path, "html")
            print(f"📄 HTML report: {html_path}")
        
        if args.format in ["markdown", "both"]:
            md_path = output_dir / "comparison_report.md"
            generate_comparison_report(comparison_results, md_path, "markdown")
            print(f"📝 Markdown report: {md_path}")
        
        # Print summary
        summary = comparison_results['summary']
        print(f"\n🏆 Overall Winner: {summary['overall_winner']}")
        print(f"📊 Benchmark Winners:")
        for benchmark, winner in summary['benchmark_winners'].items():
            print(f"   {benchmark}: {winner}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("⏹️ Comparison interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Comparison failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())