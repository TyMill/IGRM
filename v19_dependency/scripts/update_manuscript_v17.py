from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'IGRM_manuscript_v16.tex').read_text()
s=s.replace('Information Limits, State Availability, and Overlapping Policy Certification}',
'''Adaptive Mask Discovery, Information Limits, and State Availability}''')
s=s.replace('Research draft v1.6, 12 November 2026','Research draft v1.7, 12 November 2026')
start=s.index('\\noindent\\fcolorbox')
end=s.index('\\vspace{8pt}',start)
s=s[:start]+r'''\noindent\fcolorbox{black!25}{gray!8}{\parbox{\dimexpr\textwidth-2\fboxsep-2\fboxrule}{\small\textbf{Research draft---not a submission-ready article.} The adaptive fixed-family result assumes a known stationary target-state law and preallocated stream blocks. Its guarantees are derived from exact binomial bounds, alpha spending, and arrival concentration, not an optimality theorem. A strong fixed-sample CP-box baseline remains better near the risk threshold. The Mac v0.10 VOI results remain byte frozen and are not covered by the v1.7 acquisition theorem. Independent mathematical review, external validation, and a priority assessment are outstanding.}}
''' + s[end:]
start=s.index('\\begin{abstract}')
end=s.index('\\end{abstract}',start)+len('\\end{abstract}')
abstract=r'''\begin{abstract}
Deployment monitoring with scarce labels must control the conditional error
of accepted predictions while retaining nontrivial automation coverage.
We formulate safe-coverage recovery for finite-state acceptance masks under
predictable partial feedback, distributional uncertainty, random state
availability, and a hard label budget. Fisher--Rao and coordinatewise-box
certificates are compared with a decision-aware label-acquisition heuristic.
Change-of-measure arguments yield necessary information budgets, and
fixed-quota constructions yield sufficient bounds; their underlying tools
are classical. Our principal additional construction discovers a safe mask
from an overlapping finite family \emph{without knowing its risk margins}.
It uses exact binomial bounds at dyadic statewise query checkpoints, a
simultaneous error-spending schedule, and stagewise elimination with
preallocated arrival blocks. We prove pathwise budget feasibility,
false-certification probability at most $\delta_c$, margin-dependent
coverage recovery with probability at least
$1-\delta_c-\delta_r-\delta_a$, and a support-shrinkage bound on queried
labels. In a new paired five-scenario synthetic experiment (200 seeds per
scenario), the adaptive method recovers 50\% safe coverage with a mean of
1223 queried labels in a separated-risk case, whereas a stronger
nonadaptive exact-binomial control uses 4096 labels. For near-threshold
risks, however, the adaptive method returns no positive certified
coverage, while the fixed-sample control achieves 3.875\% mean coverage.
In the separately frozen eight-state Mac benchmark, decision-aware label
acquisition increases early Fisher--Rao-certified coverage under covariate
shift from 24.84\% to 40.14\%; a nongeometric CP-box certifier is
nonetheless more powerful. Our new guarantee covers the specified staged
algorithm, not the original VOI heuristic, and does not establish
asymptotic acquisition optimality or a unique advantage of Fisher--Rao
geometry.
\end{abstract}'''
s=s[:start]+abstract+s[end:]
s=s.replace(r'\section{Decision-aware label acquisition}',r'\section{Decision-aware label acquisition}\label{sec:acquisition}')
old='''The present research contributions, whose mathematical components are explicitly qualified against prior constrained sensing, are: (i) a precise joint risk--coverage certification and decision-value objective for a frozen target epoch; (ii) an implementable, auditable plug-in acquisition policy based on marginal certification gain; (iii) a specialized information-complexity characterization and an availability-constrained finite-horizon feasibility frontier, with the classical origin of their components made explicit; (iv) a constructive, uniform fixed-horizon guarantee for general finite policy families under statewise query quotas; and (v) a matching necessary--sufficient rate statement for a preidentified witness state and a finite-horizon separation from state-blind uniform querying; and (vi) a fixed-horizon overlapping-policy design which reuses state labels and guarantees mask-uniform selection under an explicit margin condition; and (vii) a paired experimental comparison against uniform, uncertainty-driven, and surrogate-optimal acquisition using two different certifiers. We state throughout which claims are mathematical consequences of known tools and which are unproven research hypotheses.'''
new=r'''Our contributions are: (i) explicit joint selective-risk and coverage certificates and a reproducible plug-in decision-value baseline; (ii) a state-availability-constrained information lower bound and finite-horizon sufficient designs, derived from acknowledged classical tools; (iii) a simultaneous certificate for overlapping mask families using shared state labels; (iv) an \emph{unknown-margin} adaptive stagewise algorithm with proven pathwise query budget, simultaneous false-certification control, conditional coverage recovery and a support-shrinkage label bound; and (v) frozen paired experiments including strong fixed-sample exact-binomial controls and informative negative results. We do not claim an optimal adaptive VOI policy, priority over combinatorial pure exploration, or an independent geometric advantage.'''
if old not in s: raise RuntimeError('contributions not found')
s=s.replace(old,new)
s=s.replace(r'\input{math/overlapping_policies_v16.tex}',r'\input{math/overlapping_policies_v16.tex}'+'\n'+r'\input{math/adaptive_unknown_margins_v17.tex}')
marker='\n\\section{Discussion}'
extra=r'''
\subsection{An independent adaptive unknown-margin experiment (v1.7)}\label{sec:adaptiveexperiment}
We evaluated the staged algorithm of Theorem~\ref{thm:adapt17} using
$K=4$, six overlapping masks, a known target law $q$, conditional Bernoulli
losses, $\tau=0.15$, $\gamma=0.25$, $H=30\,000$, and pathwise budget
$B=6000$ queried labels ($b=20\%$). For certification and arrivals we
used $\delta_c=\delta_a=0.025$; the analysis-only recovery argument sets
$\delta_r$ separately and does not assert a 95\% recovery probability
for every synthetic world. The algorithm uses the exact two-sided
Clopper--Pearson bounds at dyadic checkpoints and learns the mask margins
from queried outcomes. A deliberately conservative dyadic Hoeffding variant
is a secondary control. A \emph{state-blind dyadic ablation} uses exactly the
same staged confidence bounds, elimination rule and stopping time as the
adaptive CP method but continues querying every state while the stage is
active; thus its label difference isolates allocation rather than a change
in the certifier. The strongest comparator selects a fixed number of
query indices \emph{before observing any labels} and uses ordinary
one-sided Bonferroni Clopper--Pearson intervals at the same
false-certification allocation $\delta_c=0.025$. We also rerun these
strong fixed-sample baselines with the more permissive level $0.05$ as
an intentionally baseline-favoring sensitivity analysis.
We report both 4096 labels, matched to the adaptive scheme's
\emph{worst-case} query cap, and the full 6000-label operational budget;
this comparison does not incorrectly impose alpha spending on the
nonadaptive control. All methods see paired state/loss streams; a separate
fixed uniform-query randomization is independent of the outcomes.

Five prespecified synthetic worlds were considered: clearly separated
errors, multiple safe masks, near-threshold errors, all-unsafe masks, and
a safe mask involving a rare state. Each has 200 independently seeded
replications. True risk vectors are accessed \emph{only} by the data
generator and offline evaluation code, never by the acquisition algorithm
or its certifier. This constitutes a separate experiment from the frozen
Mac 30-seed v0.10 results.

\begin{table}[htbp]\centering\small
\caption{Adaptive unknown-margin discovery versus fixed-sample CP-box
controls (200 paired streams per world). Entries show mean certified
population coverage (\%) and queried labels. Here the fixed CP baselines
use the \emph{same} false-certification allocation $\delta_c=0.025$.
Uncertified outputs count as zero coverage. The matched fixed baseline uses
the adaptive worst-case label cap, not its smaller realized mean.}\label{tab:adaptive17}
\begin{tabular}{lrrrrr}\toprule
& \multicolumn{2}{c}{Adaptive CP} & State-blind & \multicolumn{2}{c}{Fixed CP coverage}\\
World & Coverage & Labels & Labels & 4096 & 6000 \\ \midrule
Separated risks & 50.00 & 1223 & 1623 & 50.00 & 50.00 \\
Multiple safe masks & 75.00 & 4096 & 4096 & 75.00 & 75.00 \\
Near threshold & 0.00 & 4096 & 4096 & 1.875 & 4.50 \\
All unsafe & 0.00 & 1907 & 2591 & 0.00 & 0.00 \\
Rare safe mask & 30.00 & 441 & 685 & 30.00 & 30.00 \\ \bottomrule
\end{tabular}
\end{table}

\begin{figure}[htbp]\centering
\includegraphics[width=.96\textwidth]{figures/adaptive_label_cost_v17.pdf}
\caption{Mean label consumption. Adaptive CP and fixed CP have
comparable peak certification power in separated-risk worlds, but the
adaptive procedure often stops after eliminating irrelevant candidate masks.
The full-budget control always acquires 6000 labels.}\label{fig:adaptive-label-cost}
\end{figure}
\begin{figure}[htbp]\centering
\includegraphics[width=.96\textwidth]{figures/adaptive_coverage_v17.pdf}
\caption{Mean certified population coverage. The fixed-sample CP-box
baseline is more powerful near the risk threshold, demonstrating the cost of
anytime validity and stagewise checkpointing.}\label{fig:adaptive-coverage}
\end{figure}

No unsafe certification or quota-shortage event occurred in the 7000
method-level outcomes. Repetitions sharing a stream and comparisons within
one world are dependent; the observed zero count does not establish an
exact confidence guarantee or independence across method-level records.
In the separated and rare-safe worlds, the adaptive scheme uses about
70\% and 89\% fewer labels than the matched fixed-sample baseline while
recovering the same mean oracle coverage. Relative to the stronger
\emph{equal-certifier, equal-stopping-rule} state-blind dyadic ablation,
the adaptive label-allocation saving is 24.6\% in the separated world,
26.4\% in the all-unsafe world and 35.6\% for a rare safe mask.
The adaptive and state-blind dyadic methods select the same masks and finish
at the same stage on the same synthetic streams; the saving therefore
isolates query allocation rather than earlier stopping or a favorable
confidence level. Near the frontier, the proposed
method has \emph{zero} certified coverage across the 200 tested seeds;
the same-error-level fixed CP-box control reaches 1.875\% with 4096 labels
and 4.50\% with 6000 labels. With a more permissive $0.05$ significance
budget, those fixed-sample values rise to 3.875\% and 9.25\%, respectively.
The negative conclusion survives the equal-level comparison. The primary conclusion is therefore \emph{conditional label
efficiency under easy margins}, not universal dominance or optimality.
'''
if marker not in s: raise RuntimeError('discussion marker missing')
s=s.replace(marker, '\n'+extra+marker)
s=s.replace('The v1.6 overlapping-mask result additionally proves a fixed-horizon uniform certificate, not a new general solution to combinatorial pure exploration.',
'''The additional v1.7 stagewise algorithm removes the need for supplied safety
margins and admits an explicit adaptive-pruning label bound, but remains
conservative at tight risk margins, as the stronger fixed-sample CP-box baseline
demonstrates. Neither algorithm nor theorem resolves optimal adaptive
acquisition for general overlapping policies.''')
s=s.replace('The simulator has only eight discrete states and prescribed error rates, and favorable results may depend on this model.',
'''The original simulator has only eight discrete states and prescribed error rates; the separate adaptive-mask stress test has four states and prespecified policy families. Favorable results may depend on these small models.''')
s=s.replace('No hyperparameter-stable multi-dataset evaluation has yet been completed.',
'''No independently curated industrial or medical dataset and no hyperparameter-stable multi-dataset evaluation have yet been completed.''')
s=s.replace(r'\section*{Data and Code Availability}',r'''\section*{Data and Code Availability}''')
s=s.replace('The accompanying reproducibility archive records source hashes, the full CSV hash, platform versions and test output.',
r'''The accompanying reproducibility archive records source hashes, the full CSV hash, platform versions and test output. The independent v1.7 experiment is provided as \texttt{adaptive\_overlap\_v17.csv} with its own SHA-256 checksum, seed manifest, scripts and aggregate results.''')
# Add bibliography item for adjacent 2026 feasible-arm paper (verified November 2026).
s=s.replace(r'\end{thebibliography}',
 r'''\bibitem[Cai and Kandasamy(2026)]{cai2026feasibility} Cai, T.; Kandasamy, K. Constrained best arm identification with tests for feasibility. \emph{Proc. AAAI} \textbf{2026}, \emph{40}(24), 19808--19816. \url{https://doi.org/10.1609/aaai.v40i24.39063}.
\end{thebibliography}''')
s=s.replace('\\cite{chen2017combinatorial,carlsson2024linear,das2026lagrangians,xu2024active}',
            '\\cite{chen2017combinatorial,carlsson2024linear,das2026lagrangians,xu2024active,cai2026feasibility}')
(root/'IGRM_manuscript_v17.tex').write_text(s)
print('draft bytes',len(s),'math section exists',r'\input{math/adaptive_unknown_margins_v17.tex}' in s)
