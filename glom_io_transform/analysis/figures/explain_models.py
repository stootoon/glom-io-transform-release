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

        # 24 columns so the top row can split into halves of the old 3-column
        # panel: the schematic is gone, and its 3 columns go 1.5 to the phase
        # plane (now 4.5 wide) and 0.5 to each approximation (now 2.5 each).
        #   top:    | phase (4.5) | W (2.5) | J (2.5) | U (2.5) |
        #   bottom: | conn (3) | conn_ (3) | modes (3) | hist (3) |
        gs = GridSpec(6, 24)
        fig = plt.gcf()

        top_half = slice(0, 3)
        bot_half = slice(3, 6)

        # --- Top row: Diag model logic ---
        ax_phase = fig.add_subplot(gs[top_half, 0:9])
        DiagPhase.plot(plot_data.diag, [ax_phase], **kwargs.get("phase_kwargs", {}))

        ax_approx = [fig.add_subplot(gs[top_half, sl]) for sl in
                     [slice(9, 14), slice(14, 19), slice(19, 24)]]
        DiagApprox.plot(plot_data.diag, ax_approx)

        # --- Bottom row: Free model logic ---
        ax_conn, ax_conn_ = [fig.add_subplot(gs[w]) for w in
                             [(bot_half, slice(0, 6)), (bot_half, slice(6, 12))]]
        FreeConn.plot(plot_data.free, [ax_conn, ax_conn_])

        ax_modes = [fig.add_subplot(gs[3+i, 12:18]) for i in range(3)]
        FreeConnModes.plot(plot_data.free, ax_modes)

        ax_hist = fig.add_subplot(gs[bot_half, 18:24])
        FreeConnModesHist.plot(plot_data.free, [ax_hist])

        # Return these in label order
        return {"phase": ax_phase,
                "approx_W": ax_approx[0], "approx_J": ax_approx[1], "approx_U": ax_approx[2],
                "conn": ax_conn, "conn_": ax_conn_,
                "modes": ax_modes,
                "hist": ax_hist,
                }
