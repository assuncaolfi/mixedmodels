"""Pretty-printing of fitted models."""

from __future__ import annotations


class Summary:
    def __init__(self, model):
        self.model = model

    def __repr__(self) -> str:
        return self._render()

    def __str__(self) -> str:
        return self._render()

    def _render(self) -> str:
        m = self.model
        lines = []
        lines.append(f"Mixed model fit by Laplace maximum likelihood ['{m.family.name}']")
        lines.append(f"Formula: {m.formula}")
        lines.append(f"   Data: n = {m.n_observations()}")
        lines.append(f"    logLik = {m.log_likelihood():9.4f}   deviance = {m.deviance():9.4f}")
        lines.append(
            f"       AIC = {m.aic():9.4f}        BIC = {m.bic():9.4f}   df = {m.degrees_of_freedom()}"
        )
        if not m.converged:
            lines.append(f"  Warning: optimizer reported: {m._opt_message}")
        lines.append("")
        lines.append("Random effects:")
        lines.append(f"{'Groups':<20}{'Name':<20}{'Variance':>12}{'Std.Dev.':>12}  Corr")
        for blk in m.variance_components():
            sds = blk["sd"]
            cov = blk["cov"]
            corr = blk["corr"]
            cols = blk["columns"]
            for i, col in enumerate(cols):
                lhs = f"{blk['group']:<20}" if i == 0 else f"{'':<20}"
                row = f"{lhs}{col:<20}{cov[i, i]:>12.4f}{sds[i]:>12.4f}"
                if i > 0:
                    corr_str = "  " + " ".join(f"{corr[i, j]:>6.2f}" for j in range(i))
                    row += corr_str
                lines.append(row)
        if m.family.has_dispersion:
            sigma = m.sigma()
            lines.append(f"{'Residual':<20}{'':<20}{sigma**2:>12.4f}{sigma:>12.4f}")
        lines.append(
            f"Number of obs: {m.n}, groups: "
            + ", ".join(f"{b.group_name} {b.G}" for b in m.matrices.re)
        )
        lines.append("")
        lines.append("Fixed effects:")
        wald = m.wald()
        df = wald.to_frame()
        lines.append(df.to_string(float_format=lambda x: f"{x:8.4f}"))
        return "\n".join(lines)
