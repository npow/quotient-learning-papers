"""Replace the HalfCheetah value-prediction paragraph of main.tex (tables generated from the groupsplit CSV)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import make_tables as M
P = Path(__file__).resolve().parents[2] / "paper" / "main.tex"
s = P.read_text()
a = s.index("Because $\\hat n=B$, Table~\\ref{tab:mujoco} does not certify a quotient")
b = s.index("\\subsection{End-to-end $R_{\\rm crit}$ estimation}")
new = r'''Because $\hat n=B$, Table~\ref{tab:mujoco} does not certify a quotient
size; it certifies that the eight velocity-norm bins have pairwise
distinguishable continuation laws.  Sorted by mean velocity $\ell_2$-norm the
eight HalfCheetah classes span 0.005--2.40 (a $480\times$ range) and their
discounted-return distributions differ (Kruskal--Wallis over time steps,
$H=4897$; supplement, Section~D; we do not read the accompanying $p$-value as
evidence because time steps within a trajectory are strongly dependent).

\paragraph{Do the classes carry ``multi-step information'' beyond speed?  No.}
The previous version claimed that a linear predictor on the classes
($R^2=0.163$, in-sample) beat a linear predictor on instantaneous speed
($0.082$) and concluded that the quotient retains multi-step information
beyond speed.  That comparison is not valid: the class of a step is a function
of the binned velocity norm alone ($\ell=1$, $z(o)=\|o\|_2$), so any predictor
of the classes is a piecewise-constant function of speed, and a nonlinear
function of speed can match it; the only thing the comparison showed was that
the return is a nonlinear function of speed.  We therefore re-did it with a
nonlinear speed baseline and \emph{trajectory-level} train/test splits: 5-fold
GroupKFold over trajectories, the quotient (quantile edges and classes) refitted on
training trajectories only, discounted return-to-go ($\gamma=0.99$) as target,
pooled out-of-fold $R^2$ with a trajectory-level bootstrap (2000 resamples) for
95\% intervals, and three collection seeds (1000 trajectories of 100 steps each;
mean over seeds in Table~\ref{tab:hcsplit}, file
\texttt{halfcheetah\_groupsplit\_value.csv}).

\begin{table}[t]
\centering
\small
\caption{Held-out value prediction on PO-HalfCheetah (trajectory-level 5-fold
  split; out-of-fold $R^2$, mean over 3 collection seeds, mean of the
  bootstrap 95\% interval endpoints; last column is the difference to the
  gradient-boosted speed baseline).  ``auto'': silhouette-selected $B$ (2--3
  classes per fold); ``8'': $B=8$.}
\label{tab:hcsplit}
\begin{tabular}{llllll}
\toprule
Policy & $B$ & Predictor & $R^2$ & 95\% CI & $\Delta$ vs speed (trees) \\
\midrule
@@HC@@
\bottomrule
\end{tabular}
\end{table}

Out of fold, the quotient classes are \emph{worse} than a nonlinear function of
speed under both policies and both $B$ rules (differences with intervals that
exclude $0$), adding the classes to the speed trees changes nothing
(difference $0.000$), and the classes beat the \emph{linear} speed predictor only
when $B=8$ under the sinusoidal policy ($0.256$ versus $0.084$), which is again
nonlinearity in speed.  Trees on all eight observed velocities reach
$R^2\approx0.89$ (sinusoidal) and $0.36$ (passive), so most of the predictable
return lives in directions the scalar norm discards.  We withdraw the claim
that the quotient retains multi-step information beyond speed; what the data
support is that the classes are a coarse, return-informative discretisation of
speed.  An earlier draft compared against a discretised G-test baseline said to
saturate at 64 classes; no result file for that baseline exists and we do not
report it.

'''.replace("@@HC@@", M.table_hc())
s = s[:a] + new + s[b:]
P.write_text(s)
