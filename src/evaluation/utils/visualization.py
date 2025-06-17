"""
Visualization utilities for evaluation results.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import pandas as pd
import numpy as np
from loguru import logger

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("Matplotlib/Seaborn not available - visualization will be limited")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available - interactive plots will be limited")


class EvaluationVisualizer:
    """Create visualizations for evaluation results."""
    
    def __init__(self, config: Dict):
        """Initialize visualizer."""
        self.config = config
        self.output_dir = Path(config.get("results_dir", "src/evaluation/results"))
        self.style = config.get("visualization", {}).get("style", "default")
        
        # Setup plotting style
        if PLOTTING_AVAILABLE:
            self._setup_plotting_style()
    
    def _setup_plotting_style(self):
        """Setup matplotlib/seaborn styling."""
        plt.style.use('default')
        if 'seaborn' in plt.style.available:
            plt.style.use('seaborn-v0_8')
        
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def create_comprehensive_dashboard(self, results: Dict, output_dir: Optional[Path] = None) -> Dict[str, str]:
        """Create comprehensive visualization dashboard."""
        if output_dir is None:
            output_dir = self.output_dir / "visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("📊 Creating comprehensive evaluation dashboard...")
        
        created_plots = {}
        
        # Performance comparison plots
        if PLOTLY_AVAILABLE:
            perf_plot = self._create_performance_comparison_plotly(results, output_dir)
            created_plots["performance_comparison"] = perf_plot
            
            benchmark_plot = self._create_benchmark_analysis_plotly(results, output_dir)
            created_plots["benchmark_analysis"] = benchmark_plot
            
            model_comparison = self._create_model_comparison_plotly(results, output_dir)
            created_plots["model_comparison"] = model_comparison
        
        elif PLOTTING_AVAILABLE:
            perf_plot = self._create_performance_comparison_matplotlib(results, output_dir)
            created_plots["performance_comparison"] = perf_plot
            
            benchmark_plot = self._create_benchmark_analysis_matplotlib(results, output_dir)
            created_plots["benchmark_analysis"] = benchmark_plot
        
        # Always create summary statistics (text-based)
        summary_plot = self._create_summary_statistics(results, output_dir)
        created_plots["summary_statistics"] = summary_plot
        
        logger.info(f"✅ Created {len(created_plots)} visualizations in {output_dir}")
        return created_plots
    
    def _create_performance_comparison_plotly(self, results: Dict, output_dir: Path) -> str:
        """Create interactive performance comparison using Plotly."""
        detailed_results = results.get("detailed_results", {})
        
        # Prepare data for plotting
        plot_data = []
        for benchmark_name, benchmark_data in detailed_results.items():
            models = benchmark_data.get("models", {})
            for model_name, model_data in models.items():
                if model_data.get("status") == "completed":
                    accuracy = model_data.get("performance_metrics", {}).get("overall_accuracy", 0)
                    plot_data.append({
                        "Benchmark": benchmark_name,
                        "Model": model_name,
                        "Accuracy": accuracy
                    })
        
        if not plot_data:
            return "No data available for performance comparison"
        
        df = pd.DataFrame(plot_data)
        
        # Create interactive bar plot
        fig = px.bar(
            df, 
            x="Benchmark", 
            y="Accuracy", 
            color="Model",
            title="Model Performance Comparison Across Benchmarks",
            labels={"Accuracy": "Accuracy (%)"},
            height=600
        )
        
        fig.update_layout(
            xaxis_title="Benchmark",
            yaxis_title="Accuracy (%)",
            legend_title="Model",
            hovermode='x unified'
        )
        
        # Add performance target lines
        target_lines = {
            "mirage": 70,
            "medreason": 70,
            "pubmedqa": 75,
            "msmarco": 35  # Different scale
        }
        
        for benchmark, target in target_lines.items():
            if benchmark in df["Benchmark"].values:
                fig.add_hline(
                    y=target, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"Target: {target}%"
                )
        
        # Save plot
        plot_file = output_dir / "performance_comparison.html"
        fig.write_html(plot_file)
        
        return str(plot_file)
    
    def _create_benchmark_analysis_plotly(self, results: Dict, output_dir: Path) -> str:
        """Create benchmark difficulty analysis using Plotly."""
        benchmark_analysis = results.get("benchmark_analysis", {})
        
        if not benchmark_analysis:
            return "No benchmark analysis data available"
        
        # Prepare data
        benchmarks = list(benchmark_analysis.keys())
        avg_performances = [data.get("average_performance", 0) for data in benchmark_analysis.values()]
        difficulties = [data.get("difficulty_score", 0) for data in benchmark_analysis.values()]
        model_counts = [data.get("model_count", 0) for data in benchmark_analysis.values()]
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Average Performance", "Difficulty Score", "Performance Distribution", "Model Coverage"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Average performance bar chart
        fig.add_trace(
            go.Bar(x=benchmarks, y=avg_performances, name="Avg Performance", 
                   marker_color="lightblue"),
            row=1, col=1
        )
        
        # Difficulty score
        fig.add_trace(
            go.Bar(x=benchmarks, y=difficulties, name="Difficulty", 
                   marker_color="lightcoral"),
            row=1, col=2
        )
        
        # Performance range (if available)
        performance_ranges = []
        for data in benchmark_analysis.values():
            perf_range = data.get("performance_range", {})
            performance_ranges.append({
                "min": perf_range.get("min", 0),
                "max": perf_range.get("max", 0)
            })
        
        # Performance distribution (box plot style)
        for i, (benchmark, perf_range) in enumerate(zip(benchmarks, performance_ranges)):
            fig.add_trace(
                go.Scatter(
                    x=[benchmark, benchmark],
                    y=[perf_range["min"], perf_range["max"]],
                    mode="lines+markers",
                    name=f"{benchmark} Range",
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Model coverage
        fig.add_trace(
            go.Bar(x=benchmarks, y=model_counts, name="Model Count", 
                   marker_color="lightgreen"),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Benchmark Analysis Dashboard",
            showlegend=False
        )
        
        # Save plot
        plot_file = output_dir / "benchmark_analysis.html"
        fig.write_html(plot_file)
        
        return str(plot_file)
    
    def _create_model_comparison_plotly(self, results: Dict, output_dir: Path) -> str:
        """Create model comparison radar chart using Plotly."""
        model_comparison = results.get("model_comparison", {})
        detailed_results = results.get("detailed_results", {})
        
        # Prepare data for radar chart
        models = set()
        benchmarks = set()
        
        # Collect all models and benchmarks
        for benchmark_data in detailed_results.values():
            for model_name in benchmark_data.get("models", {}):
                models.add(model_name)
            benchmarks.update(detailed_results.keys())
        
        models = list(models)
        benchmarks = list(benchmarks)
        
        if len(models) < 2:
            return "Need at least 2 models for comparison"
        
        # Create radar chart
        fig = go.Figure()
        
        colors = ["blue", "red", "green", "orange", "purple"]
        
        for i, model in enumerate(models):
            model_scores = []
            
            for benchmark in benchmarks:
                benchmark_data = detailed_results.get(benchmark, {})
                model_data = benchmark_data.get("models", {}).get(model, {})
                
                if model_data.get("status") == "completed":
                    accuracy = model_data.get("performance_metrics", {}).get("overall_accuracy", 0)
                    model_scores.append(accuracy)
                else:
                    model_scores.append(0)
            
            fig.add_trace(go.Scatterpolar(
                r=model_scores + [model_scores[0]],  # Close the polygon
                theta=benchmarks + [benchmarks[0]],
                fill='toself',
                name=model,
                line_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Model Performance Comparison (Radar Chart)",
            height=600
        )
        
        # Save plot
        plot_file = output_dir / "model_comparison_radar.html"
        fig.write_html(plot_file)
        
        return str(plot_file)
    
    def _create_performance_comparison_matplotlib(self, results: Dict, output_dir: Path) -> str:
        """Create performance comparison using matplotlib."""
        detailed_results = results.get("detailed_results", {})
        
        # Prepare data
        benchmarks = list(detailed_results.keys())
        models = set()
        
        for benchmark_data in detailed_results.values():
            models.update(benchmark_data.get("models", {}).keys())
        
        models = list(models)
        
        # Create performance matrix
        performance_matrix = np.zeros((len(models), len(benchmarks)))
        
        for i, model in enumerate(models):
            for j, benchmark in enumerate(benchmarks):
                benchmark_data = detailed_results.get(benchmark, {})
                model_data = benchmark_data.get("models", {}).get(model, {})
                
                if model_data.get("status") == "completed":
                    accuracy = model_data.get("performance_metrics", {}).get("overall_accuracy", 0)
                    performance_matrix[i, j] = accuracy
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            performance_matrix,
            xticklabels=benchmarks,
            yticklabels=models,
            annot=True,
            fmt='.1f',
            cmap='RdYlBu_r',
            center=70,
            cbar_kws={'label': 'Accuracy (%)'}
        )
        
        plt.title('Model Performance Across Benchmarks')
        plt.xlabel('Benchmarks')
        plt.ylabel('Models')
        plt.tight_layout()
        
        # Save plot
        plot_file = output_dir / "performance_heatmap.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)
    
    def _create_benchmark_analysis_matplotlib(self, results: Dict, output_dir: Path) -> str:
        """Create benchmark analysis using matplotlib."""
        benchmark_analysis = results.get("benchmark_analysis", {})
        
        if not benchmark_analysis:
            return "No benchmark analysis data"
        
        # Prepare data
        benchmarks = list(benchmark_analysis.keys())
        avg_performances = [data.get("average_performance", 0) for data in benchmark_analysis.values()]
        difficulties = [data.get("difficulty_score", 0) for data in benchmark_analysis.values()]
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Average performance
        bars1 = ax1.bar(benchmarks, avg_performances, color='skyblue', alpha=0.7)
        ax1.set_title('Average Performance by Benchmark')
        ax1.set_ylabel('Average Accuracy (%)')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars1, avg_performances):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Difficulty scores
        bars2 = ax2.bar(benchmarks, difficulties, color='lightcoral', alpha=0.7)
        ax2.set_title('Difficulty Score by Benchmark')
        ax2.set_ylabel('Difficulty Score')
        ax2.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars2, difficulties):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_dir / "benchmark_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)
    
    def _create_summary_statistics(self, results: Dict, output_dir: Path) -> str:
        """Create text-based summary statistics."""
        summary = results.get("summary", {})
        
        summary_text = "# Evaluation Summary Statistics\n\n"
        
        # Overall statistics
        summary_text += f"**Total Benchmarks:** {summary.get('total_benchmarks', 0)}\n"
        summary_text += f"**Total Models:** {summary.get('total_models', 0)}\n"
        summary_text += f"**Successful Evaluations:** {summary.get('successful_evaluations', 0)}\n"
        summary_text += f"**Failed Evaluations:** {summary.get('failed_evaluations', 0)}\n\n"
        
        # Best performers
        best_performers = summary.get("best_performers", {})
        if "overall" in best_performers:
            best = best_performers["overall"]
            summary_text += f"**Best Overall Performer:** {best.get('model', 'N/A')} ({best.get('accuracy', 0):.1f}%)\n\n"
        
        # Average performance by model
        avg_performance = summary.get("average_performance", {})
        if avg_performance:
            summary_text += "## Average Performance by Model\n\n"
            for model, perf_data in avg_performance.items():
                mean_acc = perf_data.get("mean_accuracy", 0)
                std_acc = perf_data.get("std_accuracy", 0)
                benchmarks_completed = perf_data.get("benchmarks_completed", 0)
                
                summary_text += f"**{model}:**\n"
                summary_text += f"  - Mean Accuracy: {mean_acc:.1f}% (±{std_acc:.1f}%)\n"
                summary_text += f"  - Benchmarks Completed: {benchmarks_completed}\n\n"
        
        # Benchmark analysis
        benchmark_analysis = results.get("benchmark_analysis", {})
        if benchmark_analysis:
            summary_text += "## Benchmark Difficulty Analysis\n\n"
            for benchmark, analysis in benchmark_analysis.items():
                avg_perf = analysis.get("average_performance", 0)
                difficulty_cat = analysis.get("difficulty_category", "Unknown")
                model_count = analysis.get("model_count", 0)
                
                summary_text += f"**{benchmark.upper()}:**\n"
                summary_text += f"  - Average Performance: {avg_perf:.1f}%\n"
                summary_text += f"  - Difficulty: {difficulty_cat}\n"
                summary_text += f"  - Models Evaluated: {model_count}\n\n"
        
        # Model comparison summary
        model_comparison = results.get("model_comparison", {})
        rankings = model_comparison.get("rankings", {})
        if "overall" in rankings:
            summary_text += "## Overall Model Rankings\n\n"
            for ranking in rankings["overall"]:
                rank = ranking.get("rank", 0)
                model = ranking.get("model", "Unknown")
                score = ranking.get("average_score", 0)
                summary_text += f"{rank}. **{model}** - {score:.1f}%\n"
        
        # Save summary
        summary_file = output_dir / "evaluation_summary.md"
        with open(summary_file, 'w') as f:
            f.write(summary_text)
        
        return str(summary_file)
    
    def create_comparison_plots(self, evaluation_results: Dict, output_dir: Optional[Path] = None) -> Dict[str, str]:
        """Create plots for model comparison."""
        if output_dir is None:
            output_dir = self.output_dir / "comparative"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("📈 Creating model comparison plots...")
        
        created_plots = {}
        
        if PLOTLY_AVAILABLE:
            # Performance trend plot
            trend_plot = self._create_performance_trend_plotly(evaluation_results, output_dir)
            created_plots["performance_trend"] = trend_plot
            
            # Score distribution plot
            distribution_plot = self._create_score_distribution_plotly(evaluation_results, output_dir)
            created_plots["score_distribution"] = distribution_plot
        
        # Efficiency comparison (always create)
        efficiency_plot = self._create_efficiency_comparison(evaluation_results, output_dir)
        created_plots["efficiency_comparison"] = efficiency_plot
        
        return created_plots
    
    def _create_performance_trend_plotly(self, results: Dict, output_dir: Path) -> str:
        """Create performance trend over benchmarks."""
        detailed_results = results.get("results", {})
        
        # Prepare data
        plot_data = []
        benchmark_order = ["mirage", "medreason", "pubmedqa", "msmarco"]
        
        for benchmark in benchmark_order:
            if benchmark in detailed_results:
                benchmark_results = detailed_results[benchmark]
                for model_name, model_data in benchmark_results.items():
                    if "error" not in model_data:
                        metrics = model_data.get("metrics", {})
                        accuracy = metrics.get("accuracy", 0)
                        plot_data.append({
                            "benchmark": benchmark,
                            "model": model_name,
                            "accuracy": accuracy
                        })
        
        if not plot_data:
            return "No trend data available"
        
        df = pd.DataFrame(plot_data)
        
        # Create line plot
        fig = px.line(
            df,
            x="benchmark",
            y="accuracy", 
            color="model",
            title="Performance Trend Across Benchmarks",
            markers=True,
            height=600
        )
        
        fig.update_layout(
            xaxis_title="Benchmark",
            yaxis_title="Accuracy (%)",
            legend_title="Model"
        )
        
        # Save plot
        plot_file = output_dir / "performance_trend.html"
        fig.write_html(plot_file)
        
        return str(plot_file)
    
    def _create_score_distribution_plotly(self, results: Dict, output_dir: Path) -> str:
        """Create score distribution comparison."""
        detailed_results = results.get("results", {})
        
        # Collect individual question scores
        score_data = []
        
        for benchmark_name, benchmark_results in detailed_results.items():
            for model_name, model_data in benchmark_results.items():
                if "error" not in model_data:
                    individual_results = model_data.get("individual_results", [])
                    for result in individual_results:
                        if "score" in result:
                            score_data.append({
                                "benchmark": benchmark_name,
                                "model": model_name,
                                "score": result["score"]
                            })
        
        if not score_data:
            return "No score distribution data available"
        
        df = pd.DataFrame(score_data)
        
        # Create box plot
        fig = px.box(
            df,
            x="benchmark",
            y="score",
            color="model",
            title="Score Distribution by Benchmark and Model",
            height=600
        )
        
        fig.update_layout(
            xaxis_title="Benchmark",
            yaxis_title="Score",
            legend_title="Model"
        )
        
        # Save plot
        plot_file = output_dir / "score_distribution.html"
        fig.write_html(plot_file)
        
        return str(plot_file)
    
    def _create_efficiency_comparison(self, results: Dict, output_dir: Path) -> str:
        """Create efficiency comparison chart."""
        detailed_results = results.get("results", {})
        
        # Collect efficiency data
        efficiency_data = []
        
        for benchmark_name, benchmark_results in detailed_results.items():
            for model_name, model_data in benchmark_results.items():
                if "error" not in model_data:
                    eval_time = model_data.get("evaluation_time", 0)
                    total_questions = model_data.get("total_questions", 1)
                    time_per_question = eval_time / total_questions if total_questions > 0 else 0
                    
                    efficiency_data.append({
                        "benchmark": benchmark_name,
                        "model": model_name,
                        "evaluation_time": eval_time,
                        "time_per_question": time_per_question,
                        "questions_per_minute": (total_questions / eval_time * 60) if eval_time > 0 else 0
                    })
        
        if not efficiency_data:
            return "No efficiency data available"
        
        # Create efficiency summary text
        efficiency_text = "# Efficiency Comparison\n\n"
        
        # Group by model
        df = pd.DataFrame(efficiency_data)
        
        for model in df["model"].unique():
            model_data = df[df["model"] == model]
            avg_time_per_q = model_data["time_per_question"].mean()
            avg_qpm = model_data["questions_per_minute"].mean()
            
            efficiency_text += f"## {model}\n"
            efficiency_text += f"- Average time per question: {avg_time_per_q:.2f} seconds\n"
            efficiency_text += f"- Average questions per minute: {avg_qpm:.1f}\n"
            efficiency_text += f"- Benchmarks evaluated: {len(model_data)}\n\n"
        
        # Save efficiency comparison
        efficiency_file = output_dir / "efficiency_comparison.md"
        with open(efficiency_file, 'w') as f:
            f.write(efficiency_text)
        
        return str(efficiency_file)
    
    def generate_html_report(self, results: Dict, plot_files: Dict[str, str], output_dir: Path) -> str:
        """Generate comprehensive HTML report with embedded plots."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Medical RAG Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
                .section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
                .plot-container { text-align: center; margin: 20px 0; }
                .metric { background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 3px; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
        """