from .controls import render_controls
from .optimization import render_optimization
from .scenarios import render_scenarios

def render_sidebar(disabled: bool):
    render_controls(disabled)
    render_optimization(disabled)
    render_scenarios(disabled)

