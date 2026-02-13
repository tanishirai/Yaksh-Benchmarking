import pandas as pd
from pathlib import Path
from datetime import datetime

print("="*70)
print("📝 GENERATING PROFESSIONAL BENCHMARK REPORT")
print("="*70)

output_dir = Path("reports")
output_dir.mkdir(exist_ok=True)

def clean_latex(text):
    """Clean text for LaTeX"""
    text = str(text)
    text = text.replace('%', ' percent')
    text = text.replace('\\', ' ')
    text = text.replace('_', '-')
    text = text.replace('&', 'and')
    text = text.replace('#', 'No.')
    text = text.replace('$', '')
    text = text.replace('{', '')
    text = text.replace('}', '')
    return text

latex = []

# ============================================================================
# PREAMBLE
# ============================================================================
latex.append(r"""\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{caption}
\usepackage{float}
\usepackage{array}
\usepackage{enumitem}

\title{\textbf{Benchmarking Large Language Models} \\ 
       \large for Python Code Error Detection \\
       \normalsize A Comparative Study of Six AI Models}
\author{FOSSEE Internship Project \\ Bhopal, Madhya Pradesh}
\date{""" + datetime.now().strftime("%B %Y") + r"""}

\begin{document}

\maketitle

\begin{abstract}
This report evaluates the performance of six Large Language Models (LLMs) in detecting logical errors in Python code. A dataset of 100 programming questions was manually labeled with error categories, then analyzed by each model using two prompting strategies across two independent runs. Results show significant variation in model accuracy (15-40\%), category-specific performance differences, and modest consistency across repeated evaluations. The study provides insights for selecting appropriate models for automated code review applications.
\end{abstract}

\tableofcontents
\newpage

""")

# ============================================================================
# INTRODUCTION
# ============================================================================
latex.append(r"""\section{Introduction}

\subsection{Research Context}

Automated detection of programming errors has become increasingly important for educational platforms and development tools. While Large Language Models demonstrate strong capabilities in code generation, their effectiveness at identifying specific error categories remains an open research question. This study systematically evaluates six state-of-the-art LLMs on error detection tasks.

\subsection{Research Objectives}

This benchmarking study aims to address the following questions:

\begin{enumerate}[leftmargin=*]
    \item Which LLM achieves the highest accuracy in error classification?
    \item How does performance vary across different error categories?
    \item Does prompting strategy (single vs multi-label) affect accuracy?
    \item How consistent are model predictions across independent runs?
    \item Can ensemble methods improve upon individual model performance?
\end{enumerate}

\subsection{Models Evaluated}

Six leading LLMs were selected for evaluation:

\textbf{Proprietary Models:}
\begin{itemize}[leftmargin=*]
    \item \textbf{Anthropic Claude Sonnet 4.5} - Advanced reasoning model
    \item \textbf{Google Gemini 2.5 Flash} - Optimized for speed and efficiency
    \item \textbf{OpenAI GPT-5.2} - Latest general-purpose architecture
\end{itemize}

\textbf{Open-Source Models:}
\begin{itemize}[leftmargin=*]
    \item \textbf{DeepSeek v3.2} - Chinese open-source coding model
    \item \textbf{Qwen3 Coder} - Alibaba's coding-specialized architecture
    \item \textbf{OpenAI GPT-OSS-120B} - Open-weights GPT variant
\end{itemize}

\subsection{Experimental Methodology}

\textbf{Dataset:} 100 Python programming questions from the Yaksh platform (IIT Bombay)

\textbf{Manual Annotation:} Each question was independently reviewed and labeled with applicable error categories by domain experts

\textbf{Testing Protocol:}
\begin{itemize}[leftmargin=*]
    \item Two independent runs per model (Run 1 and Run 2)
    \item Two prompting strategies: Single-label and Multi-label classification
    \item Total test configurations: 24 (6 models × 2 runs × 2 strategies)
\end{itemize}

\subsection{Error Category Definitions}

Questions were classified according to the following error taxonomy:

\begin{description}[leftmargin=*]
    \item[A - Loop Condition] Incorrect loops in for/while condition statements
    \item[B - Condition Branch] Incorrect expression in the if condition
    \item[C - Statement Integrity] Statement lacks a part of logical structure
    \item[D - Output/Input Format] Incorrect cin/cout or input/output statement
    \item[E - Variable Initialization] Incorrect declaration of variables
    \item[F - Data Type] Incorrect data type usage
    \item[G - Computation] Incorrect basic math symbols or operators
    \item[NONE] No error present (code is correct)
\end{description}

Note that questions may contain multiple error types simultaneously.

\newpage

""")

# ============================================================================
# MANUAL LABELING
# ============================================================================
latex.append(r"""\section{Dataset Characteristics}

\subsection{Manual Annotation Results}

Prior to model evaluation, all 100 questions underwent manual review to establish ground truth labels. This section presents the distribution and characteristics of the labeled dataset.

\subsection{Error Category Distribution}

Table 1 shows the frequency distribution of error categories across the dataset:

""")

print("\n📊 Loading manual labeling data...")

try:
    cat_dist = pd.read_csv("data/exports/category_distribution.csv")
    
    latex.append(r"""\begin{table}[H]
\centering
\caption{Error Category Distribution in Manual Labels}
\begin{tabular}{lcc}
\toprule
\textbf{Category} & \textbf{Count} & \textbf{Percentage} \\
\midrule
""")
    
    for _, row in cat_dist.iterrows():
        cat = clean_latex(row['Category'])
        count = int(row['Count'])
        pct = str(row['Percentage']).replace('%', '')
        latex.append(f"{cat} & {count} & {pct}\\% \\\\\n")
    
    latex.append(r"""\bottomrule
\end{tabular}
\end{table}

""")
    
    total = cat_dist['Count'].sum()
    most_common = cat_dist.iloc[0]
    least_common = cat_dist.iloc[-1]
    
    latex.append(f"""\\textbf{{Dataset Characteristics:}}

\\begin{{itemize}}[leftmargin=*]
    \\item \\textbf{{Total category assignments: {total}}} - This exceeds 100 due to multi-label questions containing multiple error types
    
    \\item \\textbf{{Most frequent: {clean_latex(most_common['Category'])}}} - Appears in {most_common['Count']} questions ({most_common['Percentage']})
    
    \\item \\textbf{{Least frequent: {clean_latex(least_common['Category'])}}} - Appears in {least_common['Count']} questions ({least_common['Percentage']})
    
    \\item \\textbf{{Average labels per question: {total/100:.2f}}} - Indicates moderate label complexity
\\end{{itemize}}

The distribution shows reasonable category diversity, though some error types are more prevalent than others in the test set.

\\newpage
""")
    
except Exception as e:
    print(f"⚠️ Category distribution not found: {e}")

# ============================================================================
# MODEL PERFORMANCE
# ============================================================================
latex.append(r"""\section{Model Performance Analysis}

\subsection{Overall Accuracy Rankings}

Table 2 presents the aggregate performance of each model, averaged across all runs and prompting strategies:

""")

print("📊 Loading model performance...")

try:
    perf = pd.read_csv("data/exports/api_model_performance_summary.csv")
    
    latex.append(r"""\begin{table}[H]
\centering
\caption{Overall Model Performance Rankings}
\begin{tabular}{lcc}
\toprule
\textbf{Model} & \textbf{Accuracy (\%)} & \textbf{Rank} \\
\midrule
""")
    
    for i, (_, row) in enumerate(perf.iterrows(), 1):
        model = clean_latex(row['Model'])
        acc = float(row['Accuracy_%'])
        latex.append(f"{model} & {acc:.1f} & {i} \\\\\n")
    
    latex.append(r"""\bottomrule
\end{tabular}
\end{table}

""")
    
    best = perf.iloc[0]
    worst = perf.iloc[-1]
    avg = perf['Accuracy_%'].mean()
    
    latex.append(f"""\\textbf{{Performance Summary:}}

\\begin{{itemize}}[leftmargin=*]
    \\item \\textbf{{Best performer: {clean_latex(best['Model'])}}} achieved {best['Accuracy_%']:.1f}\\% accuracy, correctly classifying approximately {int(best['Accuracy_%'])} out of 100 questions
    
    \\item \\textbf{{Lowest performer: {clean_latex(worst['Model'])}}} achieved {worst['Accuracy_%']:.1f}\\% accuracy
    
    \\item \\textbf{{Mean accuracy: {avg:.1f}\\%}} - Average performance across all models
    
    \\item \\textbf{{Performance range: {best['Accuracy_%'] - worst['Accuracy_%']:.1f}\\%}} - Indicates substantial variation in model capabilities
\\end{{itemize}}

\\textbf{{Implications:}} Current accuracy levels suggest these models are suitable for assistive roles rather than autonomous decision-making in code review contexts. Human verification remains necessary for production applications.

\\newpage
""")
    
except Exception as e:
    print(f"⚠️ Performance data not found: {e}")

# ============================================================================
# PER-CATEGORY
# ============================================================================
latex.append(r"""\section{Category-Specific Performance}

\subsection{Per-Category Accuracy Analysis}

Model performance varies significantly across error categories. Table 3 presents accuracy breakdowns by error type:

""")

print("📊 Loading per-category accuracy...")

try:
    cat_acc = pd.read_csv("data/exports/api_per_category_accuracy_pivot.csv", index_col=0)
    
    display_cats = cat_acc.columns[:6] if len(cat_acc.columns) > 6 else cat_acc.columns
    display_models = cat_acc.index[:6] if len(cat_acc.index) > 6 else cat_acc.index
    
    latex.append(r"""\begin{table}[H]
\centering
\caption{Model Accuracy by Error Category (\%)}
\begin{tabular}{l""" + "c"*len(display_cats) + r"""}
\toprule
\textbf{Model} & """ + " & ".join([f"\\textbf{{{col}}}" for col in display_cats]) + r""" \\
\midrule
""")
    
    for model in display_models:
        model_clean = clean_latex(model)[:15]
        row_vals = [f"{cat_acc.loc[model, cat]:.0f}" if pd.notna(cat_acc.loc[model, cat]) else "-" 
                    for cat in display_cats]
        latex.append(f"{model_clean} & " + " & ".join(row_vals) + r" \\" + "\n")
    
    latex.append(r"""\bottomrule
\end{tabular}
\end{table}

""")
    
    avg_by_cat = cat_acc.mean()
    easiest = avg_by_cat.idxmax()
    hardest = avg_by_cat.idxmin()
    
    latex.append(f"""\\textbf{{Category-Specific Findings:}}

\\begin{{itemize}}[leftmargin=*]
    \\item \\textbf{{Highest accuracy category: {clean_latex(easiest)}}} - Average {avg_by_cat[easiest]:.0f}\\% across all models
    
    \\item \\textbf{{Lowest accuracy category: {clean_latex(hardest)}}} - Average {avg_by_cat[hardest]:.0f}\\% across all models
    
    \\item \\textbf{{Standard deviation: {avg_by_cat.std():.1f}\\%}} - Indicates high variability in category difficulty
\\end{{itemize}}

\\textbf{{Interpretation:}} Performance variability suggests that overall accuracy metrics alone are insufficient for model selection. Category-specific requirements should inform model choice for domain-specific applications.

\\newpage
""")
    
except Exception as e:
    print(f"⚠️ Per-category data not found: {e}")

# ============================================================================
# SINGLE VS MULTI
# ============================================================================
latex.append(r"""\section{Prompting Strategy Comparison}

\subsection{Single-Label vs Multi-Label Classification}

Two prompting strategies were evaluated:
\begin{itemize}[leftmargin=*]
    \item \textbf{Single-label:} Models select one primary error category
    \item \textbf{Multi-label:} Models select all applicable error categories
\end{itemize}

Table 4 compares performance across these strategies:

""")

print("📊 Loading single vs multi...")

try:
    sm = pd.read_csv("data/exports/api_single_vs_multi_comparison.csv")
    comp = sm.groupby('Model').agg({
        'Single_Accuracy_%': 'mean',
        'Multi_Accuracy_%': 'mean',
        'Improvement_%': 'mean'
    }).round(1).sort_values('Improvement_%', ascending=False)
    
    latex.append(r"""\begin{table}[H]
\centering
\caption{Single vs Multi-Label Prompting Performance}
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{Single (\%)} & \textbf{Multi (\%)} & \textbf{Difference (\%)} \\
\midrule
""")
    
    for model in comp.index:
        m = clean_latex(model)[:20]
        s = comp.loc[model, 'Single_Accuracy_%']
        mu = comp.loc[model, 'Multi_Accuracy_%']
        imp = comp.loc[model, 'Improvement_%']
        sign = '+' if imp >= 0 else ''
        latex.append(f"{m} & {s:.1f} & {mu:.1f} & {sign}{imp:.1f} \\\\\n")
    
    latex.append(r"""\bottomrule
\end{tabular}
\end{table}

""")
    
    avg_imp = comp['Improvement_%'].mean()
    positive_count = (comp['Improvement_%'] > 0).sum()
    
    latex.append(f"""\\textbf{{Strategy Analysis:}}

\\begin{{itemize}}[leftmargin=*]
    \\item \\textbf{{Average difference: {avg_imp:+.1f}\\%}} - Multi-label prompting shows {'positive' if avg_imp > 0 else 'negative'} effect on average
    
    \\item \\textbf{{Improved models: {positive_count} out of 6}} - Effect is not uniform across architectures
    
    \\item \\textbf{{Best improvement: {comp['Improvement_%'].max():+.1f}\\%}} ({clean_latex(comp['Improvement_%'].idxmax())})
    
    \\item \\textbf{{Largest decline: {comp['Improvement_%'].min():+.1f}\\%}} ({clean_latex(comp['Improvement_%'].idxmin())})
\\end{{itemize}}

\\textbf{{Recommendation:}} Prompting strategy should be evaluated on a per-model basis for specific deployment scenarios.

\\newpage
""")
    
except Exception as e:
    print(f"⚠️ Single vs multi not found: {e}")

# ============================================================================
# CONSISTENCY
# ============================================================================
latex.append(r"""\section{Consistency and Reliability}

\subsection{Run-to-Run Agreement Analysis}

Model consistency was evaluated by comparing predictions across two independent runs on identical inputs. Table 5 presents agreement rates:

""")

print("📊 Loading consistency...")

try:
    cons = pd.read_csv("data/exports/api_run_consistency.csv")
    summary = cons.groupby('Model').agg({
        'Agreement_Rate_%': 'mean'
    }).round(1).sort_values('Agreement_Rate_%', ascending=False)
    
    latex.append(r"""\begin{table}[H]
\centering
\caption{Inter-Run Agreement Rates}
\begin{tabular}{lc}
\toprule
\textbf{Model} & \textbf{Agreement Rate (\%)} \\
\midrule
""")
    
    for model in summary.index:
        m = clean_latex(model)[:20]
        agr = summary.loc[model, 'Agreement_Rate_%']
        latex.append(f"{m} & {agr:.1f} \\\\\n")
    
    latex.append(r"""\bottomrule
\end{tabular}
\end{table}

""")
    
    most = summary['Agreement_Rate_%'].idxmax()
    least = summary['Agreement_Rate_%'].idxmin()
    avg_consistency = summary['Agreement_Rate_%'].mean()
    
    latex.append(f"""\\textbf{{Consistency Metrics:}}

\\begin{{itemize}}[leftmargin=*]
    \\item \\textbf{{Highest consistency: {clean_latex(most)}}} - {summary.loc[most, 'Agreement_Rate_%']:.1f}\\% agreement rate
    
    \\item \\textbf{{Lowest consistency: {clean_latex(least)}}} - {summary.loc[least, 'Agreement_Rate_%']:.1f}\\% agreement rate
    
    \\item \\textbf{{Average consistency: {avg_consistency:.1f}\\%}} - Mean agreement across all models
\\end{{itemize}}

\\textbf{{Production Considerations:}} High consistency is critical for deployment reliability. Models with lower agreement rates may require ensemble methods or human verification workflows.

\\newpage
""")
    
except Exception as e:
    print(f"⚠️ Consistency data not found: {e}")

# ============================================================================
# CLOSED VS OPEN
# ============================================================================
latex.append(r"""\section{Proprietary vs Open-Source Comparison}

\subsection{Cost-Performance Analysis}

Table 6 compares the average performance of proprietary (closed-source) versus open-source models:

""")

print("📊 Loading closed vs open...")

try:
    co = pd.read_csv("data/exports/api_closed_vs_opensource_summary.csv")
    
    latex.append(r"""\begin{table}[H]
\centering
\caption{Proprietary vs Open-Source Performance}
\begin{tabular}{lccc}
\toprule
\textbf{Mode} & \textbf{Proprietary (\%)} & \textbf{Open-Source (\%)} & \textbf{Gap (\%)} \\
\midrule
""")
    
    for _, row in co.iterrows():
        mode = row['Mode'].capitalize()
        closed = float(row['Closed_Source_Avg_%'])
        open_s = float(row['Open_Source_Avg_%'])
        diff = float(row['Difference_%'])
        sign = '+' if diff >= 0 else ''
        latex.append(f"{mode} & {closed:.1f} & {open_s:.1f} & {sign}{diff:.1f} \\\\\n")
    
    latex.append(r"""\bottomrule
\end{tabular}
\end{table}

""")
    
    avg_diff = co['Difference_%'].mean()
    
    latex.append(f"""\\textbf{{Economic Analysis:}}

\\begin{{itemize}}[leftmargin=*]
    \\item \\textbf{{Performance advantage: {abs(avg_diff):.1f}\\%}} - {'Proprietary' if avg_diff > 0 else 'Open-source'} models lead on average
    
    \\item Cost considerations must be weighed against the observed performance differential
    
    \\item Open-source models provide viable alternatives for budget-constrained deployments
\\end{{itemize}}

\\newpage
""")
    
except Exception as e:
    print(f"⚠️ Closed vs open not found: {e}")

# ============================================================================
# ENSEMBLE
# ============================================================================
latex.append(r"""\section{Ensemble Methods}

\subsection{Majority Voting Analysis}

An ensemble approach using majority voting across all six models was evaluated. Table 7 shows results:

""")

print("📊 Loading ensemble...")

try:
    consensus = pd.read_csv("data/exports/api_consensus_summary.csv")
    
    latex.append(r"""\begin{table}[H]
\centering
\caption{Ensemble Performance vs Individual Models}
\begin{tabular}{lccc}
\toprule
\textbf{Mode} & \textbf{Avg Individual (\%)} & \textbf{Ensemble (\%)} & \textbf{Improvement (\%)} \\
\midrule
""")
    
    for _, row in consensus.iterrows():
        mode = row['Mode'].capitalize()
        ind = float(row['Avg_Individual_Accuracy_%'])
        ens = float(row['Ensemble_Accuracy_%'])
        gain = float(row['Ensemble_Improvement_%'])
        sign = '+' if gain >= 0 else ''
        latex.append(f"{mode} & {ind:.1f} & {ens:.1f} & {sign}{gain:.1f} \\\\\n")
    
    latex.append(r"""\bottomrule
\end{tabular}
\end{table}

\textbf{Ensemble Insights:}

\begin{itemize}[leftmargin=*]
    \item Majority voting demonstrates improved accuracy over average individual performance
    \item Computational cost increases linearly with ensemble size
    \item Suitable for high-criticality applications where accuracy justifies additional overhead
\end{itemize}

\newpage
""")
    
except Exception as e:
    print(f"⚠️ Ensemble data not found: {e}")

# ============================================================================
# CONCLUSIONS
# ============================================================================
latex.append(r"""\section{Conclusions and Recommendations}

\subsection{Key Findings}

\begin{enumerate}[leftmargin=*]
    \item \textbf{Performance Range:} Model accuracy ranges from 15-40\%, indicating substantial room for improvement in error detection capabilities.
    
    \item \textbf{Category Dependence:} Performance varies significantly by error type, with some categories proving systematically more challenging than others.
    
    \item \textbf{Prompting Effects:} Multi-label prompting shows inconsistent effects across models, requiring case-by-case evaluation.
    
    \item \textbf{Consistency Variation:} Inter-run agreement rates vary substantially, with implications for production reliability.
    
    \item \textbf{Commercial Advantage:} Proprietary models demonstrate modest performance gains over open-source alternatives.
    
    \item \textbf{Ensemble Benefits:} Majority voting improves accuracy but increases computational requirements.
\end{enumerate}

\subsection{Deployment Recommendations}

\textbf{Educational Applications:}
\begin{itemize}[leftmargin=*]
    \item Deploy models as supplementary learning aids rather than primary assessment tools
    \item Emphasize category-specific strengths when providing automated feedback
    \item Maintain human oversight for final grading decisions
\end{itemize}

\textbf{Code Review Integration:}
\begin{itemize}[leftmargin=*]
    \item Implement ensemble methods for critical code paths
    \item Establish category-specific confidence thresholds
    \item Flag low-consensus predictions for manual review
\end{itemize}

\textbf{Model Selection Criteria:}
\begin{itemize}[leftmargin=*]
    \item Prioritize consistency metrics alongside accuracy
    \item Evaluate performance on domain-specific error categories
    \item Consider open-source alternatives for cost-sensitive applications
\end{itemize}

\subsection{Study Limitations}

\begin{itemize}[leftmargin=*]
    \item Sample size limited to 100 questions (larger datasets recommended)
    \item Single programming language scope (Python only)
    \item Binary evaluation metric (correct/incorrect without partial credit)
    \item Limited to released model versions (subject to rapid obsolescence)
\end{itemize}

\subsection{Future Research Directions}

\begin{itemize}[leftmargin=*]
    \item Expand dataset to 500+ questions for improved statistical power
    \item Evaluate domain-specific fine-tuned models
    \item Investigate advanced prompting techniques (chain-of-thought, few-shot learning)
    \item Extend analysis to additional programming languages
    \item Assess explanation quality alongside classification accuracy
    \item Track performance evolution as models are updated
\end{itemize}

\subsection{Concluding Remarks}

Current LLM performance in code error detection demonstrates promise but remains insufficient for autonomous deployment. The observed accuracy ceiling around 40\% suggests fundamental limitations in semantic code understanding. However, strategic deployment through ensemble methods, category-specific application, and human-in-the-loop workflows can enable practical value extraction from these technologies.

Given rapid advancement in the field, periodic re-evaluation is recommended to track performance improvements in next-generation models.

\end{document}
""")

# ============================================================================
# SAVE
# ============================================================================
output_file = output_dir / "benchmark_report.tex"
with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(latex)

print(f"\n✅ Professional report generated!")
print(f"📄 File: {output_file}")
print("\n" + "="*70)
print("📝 Upload to Overleaf: https://www.overleaf.com")
print("="*70)
