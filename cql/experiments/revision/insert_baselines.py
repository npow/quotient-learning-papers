"""Replace the old Baselines + Ablation subsections of main.tex by the new baselines comparison."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import make_tables as M
P = Path(__file__).resolve().parents[2] / "paper" / "main.tex"
s = P.read_text()
A = "%%BEGIN-BASELINES"; Bm = "%%END-BASELINES\n"
block = r'''%%BEGIN-BASELINES
\subsection{Head-to-head comparison with related methods}
\label{sec:baselines}
We compare partition accuracy (ARI against the latent state, 5 seeds, the same
trajectories for every method, no method given $n$) with re-implementations of
the closest methods (details and what is and is not faithful in
Appendix~\ref{sec:baseline-details}):
an \textbf{RKHS $\varepsilon$-machine} \citep{Brodu2022} (regularised conditional
mean embeddings of the continuation given the present observation, similarity
matrix, diffusion-map coordinates, DBSCAN or $k$-means/silhouette);
\textbf{causal-state learning} \citep{Zhang2019causal} (a GRU encoder over the
\emph{full} history trained to predict the next observation and reward, $k$-means
discretisation); a \textbf{Ferns bisimulation metric} \citep{Ferns2004,Ferns2011}
on the empirical chain over the $B=8$ quantile bins; \textbf{DBC}
\citep{Zhang2020} with its standard objective (bisimulation loss with a learned
Gaussian latent dynamics model, $k$-means/silhouette); and \textbf{multivariate
$W_1$} between bin-conditional continuation vectors (exact optimal transport),
with merge thresholds $\varepsilon$ or silhouette-selected average linkage.  The
previous version's Wasserstein baseline (first-coordinate projections) and DBC
baseline (a custom Wasserstein-matching objective) are replaced; their old rows
are removed from Table~\ref{tab:ablation}.  Bisimulation methods need a reward:
Tiger uses its native reward and the reward-free K-cycle/GaussMix use the next
observation as an emission reward, which is our choice.

\begin{table}[t]
\centering
\scriptsize
\caption{Mean ARI over 5 seeds (range of $\hat n$ in brackets) against the
  latent state.  K$k$ = K-cycle with $K=k$, G$k$ = GaussMix with $N_s=k$.
  Latent indices: Tiger 2, K2 2, K4 4, K8 8, G3 3, G5 5.  ARI is bounded
  above by a bin-readout ceiling for the binned methods (Table~\ref{tab:ari});
  the GRU conditions on the full history and is not.  Source:
  \texttt{partition\_baselines.csv}, \texttt{partition\_accuracy.csv}.}
\label{tab:baselines}
\begin{tabular}{lcccccc}
\toprule
Method & Tiger & K2 & K4 & K8 & G3 & G5 \\
\midrule
@@BASE@@
\bottomrule
\end{tabular}
\end{table}

\paragraph{What the comparison shows.}
\emph{No method is uniformly best, and our estimator is not the best baseline
on the hardest environments.}  (i) The GRU causal-state learner, which sees the
whole history, is the best method on K-cycle ($0.92$ at $K=4$, $0.86$ at $K=8$,
above the one-step-key ceiling) because the phase is determined by the history;
its $k$-means step over-counts classes ($\hat n=5$ at $K=4$, $10$--$12$ at
$K=8$) and it fails on the GaussMix random walk with $N_s=5$ ($0.35$, $\hat n=2$).
(ii) The RKHS $\varepsilon$-machine with $k$-means/silhouette is competitive with
ours where classes are well separated (Tiger $0.91$, K2 $0.97$, G3 $0.99$, G5
$0.99$ with the correct count) and weaker on K4/K8; with DBSCAN at our scaling
of its threshold it returns one cluster on Tiger, K4, K8 --- the threshold is
exactly the kind of unvalidated scale parameter that the test-based formulation
avoids, but we also did not tune it.  (iii) A Ferns metric on the bins and
silhouette-clustered $W_1$ at $B=8$ do better than our estimator at the same
$B=8$ on K4 ($0.64$ and $0.61$ versus $0.40$) and tie on Tiger, but fail at K8
($0.12$, $0.20$ versus $0.31$).  (iv) No single $W_1$ threshold works: $\varepsilon=0.5$ is right
on Tiger and K2 and collapses K4/K8 to one class; $\varepsilon\le0.2$ over-splits
Tiger.  (v) DBC fails on Tiger ($\hat n=11$--$12$, $0.16$) and is good on K2 and
G5 ($0.98$, $\hat n=5$--$6$).
The previous statement that permutation calibration is ``the only method correct
on both'' environments is withdrawn: with a validated count and $B$ chosen
without $n$, our estimator is correct on Tiger, K2, K3 and GaussMix, and other
methods match or beat it on specific environments.  What distinguishes the
estimator is not accuracy but form: it outputs a partition together with a
test-based decision for each merge, a stated target ($\Qstar_B$), and (for
variant~(T)) a sample bound, none of which the compared methods provide.

\begin{table}[t]
\centering
\caption{G-test baselines of the companion paper (finite-alphabet estimator
  after quantile discretisation of the continuation), counts of seeds with
  $\hat n=n$ (from \texttt{ablation\_sample\_efficiency.csv}; Tiger: $N=5000$, 3
  seeds; K-cycle: $K=4$, $N=3000$, $B=K$, 3 seeds).  The Wasserstein and DBC rows
  of the previous version are superseded by Table~\ref{tab:baselines}.}
\label{tab:ablation}
\begin{tabular}{lrr}
\toprule
Method & Tiger & K-cycle \\
\midrule
G-test 8-bin                               & 2/3   & 0/3 ($\hat n=8$) \\
G-test 3-bin                               & 0/3   & 0/3 ($\hat n=3$) \\
\bottomrule
\end{tabular}
\end{table}
%%END-BASELINES
'''
block = block.replace("@@BASE@@", M.table_baselines())
if A in s:
    i, j = s.index(A), s.index(Bm) + len(Bm)
    s = s[:i] + block + s[j:]
else:
    a = s.index("\\subsection{Baselines}")
    b = s.index("\\subsection{Ground-truth recovery at the aligned resolution}")
    s = s[:a] + s[b:]            # drop old baseline descriptions
    a = s.index("\\subsection{Ablation against baselines}")
    b = s.index("\\subsection{PO-MuJoCo: quotient at a chosen resolution}")
    s = s[:a] + block + "\n" + s[b:]
P.write_text(s)
