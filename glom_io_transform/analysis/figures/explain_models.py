"""The explain_models figure: Diag quartic geometry (top row) and Free
connectivity theory (bottom row). Pure orchestration -- the panels live in
figures.diag and figures.free, and consume plot_data.diag / plot_data.free."""
from .figures import plt, GridSpec
from .figures import Figure

from .diag import DiagPhase, DiagApprox
from .free import FreeConn, FreeConnModes, FreeConnModesHist


class Main(Figure):
    @classmethod
    def plot(cls, plot_data, **kwargs):
        print("PLOTTING FIGURE ExplainModels")

        # One gridspec per row rather than one grid of many columns. The
        # schematic is gone, and its width goes 1.5 panels to the phase plane
        # and 0.5 to each approximation -- ratios that are not whole columns of
        # a 12-wide grid, and that a 24-wide grid expresses only by making every
        # column too narrow for tight_layout to fit the axis labels into.
        # width_ratios states them directly and keeps the columns wide.
        #   top:    | phase (4.5) | W (2.5) | J (2.5) | U (2.5) |
        #   bottom: | conn (3) | conn_ (3) | modes (3) | hist (3) |
        fig = plt.gcf()
        outer = GridSpec(2, 1, figure=fig)
        top = outer[0].subgridspec(1, 4, width_ratios=[4.5, 2.5, 2.5, 2.5])
        # Three rows so the modes panel can stack; the others span all of them.
        bot = outer[1].subgridspec(3, 4)

        # --- Top row: Diag model logic ---
        ax_phase = fig.add_subplot(top[0, 0])
        DiagPhase.plot(plot_data.diag, [ax_phase], **kwargs.get("phase_kwargs", {}))

        ax_approx = [fig.add_subplot(top[0, i]) for i in (1, 2, 3)]
        DiagApprox.plot(plot_data.diag, ax_approx)

        # --- Bottom row: Free model logic ---
        ax_conn, ax_conn_ = [fig.add_subplot(bot[:, j]) for j in (0, 1)]
        FreeConn.plot(plot_data.free, [ax_conn, ax_conn_])

        ax_modes = [fig.add_subplot(bot[i, 2]) for i in range(3)]
        FreeConnModes.plot(plot_data.free, ax_modes)

        ax_hist = fig.add_subplot(bot[:, 3])
        FreeConnModesHist.plot(plot_data.free, [ax_hist])

        # Return these in label order
        return {"phase": ax_phase,
                "approx_W": ax_approx[0], "approx_J": ax_approx[1], "approx_U": ax_approx[2],
                "conn": ax_conn, "conn_": ax_conn_,
                "modes": ax_modes,
                "hist": ax_hist,
                }
