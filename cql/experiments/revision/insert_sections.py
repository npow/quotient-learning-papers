"""Insert the partition-accuracy and variant-T subsections (tables generated from CSVs) into paper/main.tex.
Idempotent: replaces the block between the BEGIN/END markers if present."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import make_tables as M
P = Path(__file__).resolve().parents[2] / "paper" / "main.tex"
s = P.read_text()
block = r'''%%BEGIN-REVISION-SECTIONS
\subsection{Partition-level accuracy without knowledge of $n$}
\label{sec:partition}
Table~\ref{tab:layer1} counts $\hat n=n$ at $B=n$, which uses the answer.  We
therefore score the \emph{partition}: the adjusted Rand index (ARI) between the
estimated class of every step (the class of its bin) and the latent state at
that step, over all steps with a full continuation window (5 seeds per cell,
same configurations as Table~\ref{tab:layer1}, procedure~(P)).  Two
selection rules for $B$ do not use $n$: silhouette auto-selection
(Algorithm~2, $B\in[2,32]$, observations only) and the package default
$B=8$; $B=n$ is shown as a reference.  Because a bin is a noisy readout of the
latent state, no partition of the bins can reach ARI $=1$; the last column gives
the best ARI over all merges of the $B=8$ bins (each bin mapped to its majority
latent class).

\begin{table}[t]
\centering
\small
\caption{Partition accuracy of procedure~(P) (ARI vs.\ latent state; $\hat n$
  is the range over 5 seeds; mean ARI). ``Ceiling'' is the best ARI attainable
  by merging the $B=8$ bins.  Source: \texttt{partition\_accuracy.csv}.}
\label{tab:ari}
\begin{tabular}{lrcccccccc}
\toprule
 & & \multicolumn{2}{c}{$B=n$ (reference)} & \multicolumn{3}{c}{silhouette $B$} & \multicolumn{3}{c}{$B=8$} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
Environment & $n$ & $\hat n$ & ARI & $B$ & $\hat n$ & ARI & $\hat n$ & ARI & ceiling \\
\midrule
@@ARI@@
\bottomrule
\end{tabular}
\end{table}

Three observations.  (i) Selecting $B$ without $n$ is the weak point.
Silhouette recovers the latent index and partition on Tiger, K-cycle $K\le3$ and
all GaussMix runs, but selects $B=2$ on K-cycle for $K\ge4$ ($\hat n=2$, ARI
$0.19$--$0.41$); the statement in the previous version that silhouette chose
$B=K$ on K-cycle had no result file behind it and is withdrawn.  Fixed $B=8$
recovers only Tiger and $K=2$; elsewhere it over-splits ($\hat n=5$ at $K=3$,
$8$ at $K\ge4$, $4$--$8$ on GaussMix).  (ii) At the aligned resolution
$B=n$, $\hat n=n$ is accompanied by an ARI equal to the best attainable by any
merging of those $n$ bins on every environment, i.e.\ the merge step is correct, but the ARI itself is far below 1
when noise is large ($0.31$ at $K=8$, where adjacent means are $1.25\sigma$
apart): ARI measures bin quality and merging jointly.  

\subsection{Experiments under the analysed variant (T)}
\label{sec:variantT}
Theorem~\ref{thm:main} covers variant~(T) (Section~\ref{sec:method}), so we
re-ran the ground-truth experiments with it: bins and bandwidth from a design
half, one window per trajectory and bin from the other half, independent sides,
analytic threshold with $\delta=0.05$ (5 seeds per cell, same data as
Table~\ref{tab:ari}).

\begin{table}[t]
\centering
\small
\caption{Procedure~(P) versus variant~(T): range of $\hat n$ over 5 seeds and
  mean ARI, at $B=n$ and at $B=8$.  Source: \texttt{partition\_accuracy.csv}.}
\label{tab:variantT}
\begin{tabular}{lrcccccccc}
\toprule
 & & \multicolumn{4}{c}{$B=n$} & \multicolumn{4}{c}{$B=8$} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}
 & & \multicolumn{2}{c}{(P)} & \multicolumn{2}{c}{(T)} & \multicolumn{2}{c}{(P)} & \multicolumn{2}{c}{(T)} \\
Environment & $n$ & $\hat n$ & ARI & $\hat n$ & ARI & $\hat n$ & ARI & $\hat n$ & ARI \\
\midrule
@@T@@
\bottomrule
\end{tabular}
\end{table}

\paragraph{Findings.}
(i) At the aligned resolution $B=n$, (T) returns the same $\hat n$ as (P) on
K-cycle ($K\le8$) and GaussMix $N_s\le5$ (ARI within $0.02$), and loses a class
on some GaussMix $N_s=6$ seeds ($\hat n=5$ on 3/5 seeds); on Tiger it returns
$\hat n=1$ (see below).  (ii) At $B=8$ the two procedures err in opposite
directions.  (T) over-splits less on K-cycle $K=4$ ($\hat n\in\{4,5\}$ versus $8$)
and is equal to (P) at $K=3$ and $K=8$, but its analytic threshold is
conservative and merges real classes: $\hat n=1$ on Tiger and GaussMix $N_s=3$
and $1$--$4$ on the other GaussMix runs.  Tiger at $N=3000$ is not resolved by (T) at either $B$
($\hat n=1$): with $m\approx700$ samples per side the threshold is
$t_{\alpha_0}\approx0.2$--$0.3$, above the empirical MMD between the two Tiger
continuation laws.  Scaling the sample size (\texttt{tiger\_T\_scaling.csv}, 5 seeds, $B\in\{2,8\}$): $\hat n=1$ at $N=3000$
(median threshold $0.21$ at $B=2$ and $0.26$ at $B=8$, above the largest observed
statistic, $0.17$ and $0.23$), and $\hat n=2$ with ARI $0.91$ on all 5 seeds at
both $B$ for $N=10^4$ (thresholds $0.11$--$0.14$) and $N=3\cdot10^4$ ($0.065$--$0.082$).
So (T) does resolve Tiger, with about $3$--$10\times$ the trajectories (P) needs.
(iii) Permutation calibration and BH on (T)'s data pipeline
(P$^{\rm fix}$: fixed bins and kernel, i.i.d.\ single windows, independent sides)
reproduces (P)'s output on Tiger, K-cycle $K\in\{2,4\}$ and GaussMix $N_s=4$
(same $\hat n$ on 37 of 40 comparable (seed, $B$-rule) pairs; the 3 differences are all GaussMix $N_s=4$ at $B=8$, $\hat n\in\{4,5\}$),
so the over-splitting of (P) at $B=8$ is attributable to calibration (permutation
$p$-values at a floor of $1/(R{+}1)$ with BH), not to the overlapping windows or the
per-pair bandwidth, at least in these environments.  (iv) Dec-Tiger:
we use \texttt{per\_traj} sampling (one random time per trajectory, so the
sampled law is exactly the pooled bin law; the state is constant within an
episode, and the \texttt{per\_bin} scheme would re-weight the bins' laws
toward rare visits), $B=4$, seeds $0$--$9$ (\texttt{dec\_tiger\_variant\_t.csv}).
Symmetric $(0.5,0.5)$: (T) returns $\hat n_A=\hat n_B=2$ on 10/10 seeds at $N=1000$
($\hat R_{\rm crit}=0.307$, mean absolute error $0.044$ against $0.262$) and at
$N=20000$ ($0.264$, mean absolute error $0.002$): no agent-A over-splitting,
unlike (P) at $\alpha=0.05$ (5/10 seeds, Table~\ref{tab:rcrit}).  Asymmetric
$(0.1,1.0)$: $\hat n_A=2$ on all seeds, but agent~B ($\sigma_B=1$) collapses to one
class at $N=1000$ ($\hat R_{\rm crit}=0.999$ versus $0.631$) and returns $3$--$4$ classes at
$N=20000$ ($0.563$, mean absolute error $0.068$).  Permutation+BH on the same
data pipeline (P$^{\rm fix}$) over-splits agent~A on 2/10 (symmetric) and 3/10
(asymmetric) seeds at $N=1000$ and returns $\hat n_A=\hat n_B=4$ on all 3 seeds run at
$N=20000$ (symmetric: $\hat R_{\rm crit}=1.24$ against $0.262$; only 3 seeds because each
takes about two minutes).  The over-splitting of $R_{\rm crit}$ under (P) is thus again a
property of the permutation/BH calibration: consistent with its resolving
within-class bin differences ever more finely as $N$ grows (the bin-level quotient
$\Qstar_B$ is generically finer than the latent one), while (T) stays at the
latent index but is conservative.
\textbf{Reading.}  The Section~5 results for (P) are therefore not guarantees: the
guaranteed procedure is far more conservative and needs roughly $10\times$ more
trajectories on Tiger to resolve the same two classes.  We report both because
(P) is the procedure that is practical at $N\le3000$.
%%END-REVISION-SECTIONS
'''
block = block.replace("@@ARI@@", M.table_ari()).replace("@@T@@", M.table_T())
a, b = "%%BEGIN-REVISION-SECTIONS", "%%END-REVISION-SECTIONS\n"
if a in s:
    i, j = s.index(a), s.index(b) + len(b)
    s = s[:i] + block + s[j:]
else:
    k = s.index("\\subsection{Ablation against baselines}")
    s = s[:k] + block + "\n" + s[k:]
P.write_text(s)
