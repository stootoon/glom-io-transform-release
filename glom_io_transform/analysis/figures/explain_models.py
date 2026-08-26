"""The explain_models figure: Diag quartic geometry (top row) and Free
connectivity theory (bottom row). Pure orchestration -- the panels live in
figures.diag and figures.free, and consume plot_data.diag / plot_data.free."""
import os

from .figures import plt, GridSpec
from .figures import Figure, Schem
from .figures import paths

from .diag import DiagPhase, DiagApprox
from .free import FreeConn, FreeConnModes, FreeConnModesHist


class Main(Figure):
    @classmethod
    def plot(cls, plot_data, **kwargs):
        print("PLOTTING FIGURE ExplainModels")
        art_path = os.path.join(paths.proj_path, "art")

        # 2 x 12 layout (6 rows so the modes panel can stack 3 axes):
        #   top:    | geom (3) | phase (3) | W (2) | J (2) | U (2) |
        #   bottom: | conn (3) | conn_ (3) | modes (3) | hist (3)  |
        gs = GridSpec(6, 12)
        fig = plt.gcf()

        top_half = slice(0, 3)
        bot_half = slice(3, 6)

        # --- Top row: Diag model logic ---
        ax_geom = fig.add_subplot(gs[top_half, 0:3])
        geom_art = os.path.join(art_path, "diag_geometry.png")
        if os.path.exists(geom_art):
            Schem.plot(plot_data, [ax_geom], art_file=geom_art)
        else:
            print(f"Geometry schematic not found at {geom_art}; leaving panel blank.")
            ax_geom.axis("off")

        ax_phase = fig.add_subplot(gs[top_half, 3:6])
        DiagPhase.plot(plot_data.diag, [ax_phase], **kwargs.get("phase_kwargs", {}))

        ax_approx = [fig.add_subplot(gs[top_half, sl]) for sl in
                     [slice(6, 8), slice(8, 10), slice(10, 12)]]
        DiagApprox.plot(plot_data.diag, ax_approx)

        # --- Bottom row: Free model logic ---
        ax_conn, ax_conn_ = [fig.add_subplot(gs[w]) for w in
                             [(bot_half, slice(0, 3)), (bot_half, slice(3, 6))]]
        FreeConn.plot(plot_data.free, [ax_conn, ax_conn_])

        ax_modes = [fig.add_subplot(gs[3+i, 6:9]) for i in range(3)]
        FreeConnModes.plot(plot_data.free, ax_modes)

        ax_hist = fig.add_subplot(gs[bot_half, 9:12])
        FreeConnModesHist.plot(plot_data.free, [ax_hist])

        # Return these in label order
        return {"geom": ax_geom,
                "phase": ax_phase,
                "approx_W": ax_approx[0], "approx_J": ax_approx[1], "approx_U": ax_approx[2],
                "conn": ax_conn, "conn_": ax_conn_,
                "modes": ax_modes,
                "hist": ax_hist,
                }
