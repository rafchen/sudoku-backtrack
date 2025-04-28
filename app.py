import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sudoku Constraint Satisfaction", layout="wide")

def solve_sudoku_steps(initial_grid, use_forward_checking, use_mrv, use_lcv):
    """Generates steps for solving a Sudoku puzzle via backtracking."""
    grid = initial_grid.copy()
    domains = {(r, c): ([int(grid[r, c])] if grid[r, c] else list(range(1, 10)))
               for r in range(9) for c in range(9)}
    assignment_count = 0
    backtrack_count = 0

    peers = {}
    for r in range(9):
        for c in range(9):
            br, bc = 3 * (r // 3), 3 * (c // 3)
            peer_set = (
                {(r, j) for j in range(9) if j != c} |
                {(i, c) for i in range(9) if i != r} |
                {(i, j) for i in range(br, br + 3) for j in range(bc, bc + 3) if (i, j) != (r, c)}
            )
            peers[(r, c)] = peer_set

    def pick_cell():
        """Selects the next cell to assign, optionally using MRV."""
        unassigned = [(len(domains[cell]), cell) for cell in domains if len(domains[cell]) > 1]
        if not unassigned:
            return None
        if use_mrv:
            unassigned.sort()
        return unassigned[0][1]

    def order_values(cell):
        """Orders values for a cell, optionally using LCV."""
        vals = domains[cell]
        if not use_lcv:
            return vals
        scores = []
        for v in vals:
            conflicts = sum(v in domains[p] and len(domains[p]) > 1 for p in peers[cell])
            scores.append((conflicts, v))
        return [v for _, v in sorted(scores)]

    def capture_state(event, cell=None, val=None, pruned_cells=None):
        """Captures the current solver state for visualization."""
        current_board = np.zeros((9, 9), int)
        for (r, c), dom in domains.items():
            if len(dom) == 1:
                current_board[r, c] = dom[0]

        messages = {
            "Start": "Solver initialized.",
            "Select": f"Selected cell {cell} (MRV={'On' if use_mrv else 'Off'}).",
            "Assign": f"Trying value {val} at {cell} (LCV={'On' if use_lcv else 'Off'}).",
            "Prune": f"Forward checking from {cell}={val} pruned {len(pruned_cells or [])} peer domains.",
            "Backtrack": f"Failed assignment {val} at {cell}. Backtracking.",
            "Solved": "🎉 Puzzle Solved!",
            "Failed": "❌ No solution found."
        }
        return {
            'grid': current_board,
            'domains': {c: domains[c].copy() for c in domains},
            'assignments': assignment_count,
            'backtracks': backtrack_count,
            'note': messages.get(event, f'Unknown event: {event}'),
            'current_cell': cell,
            'action': event,
            'affected_cells': pruned_cells or []
        }

    def backtrack():
        """Recursive backtracking search function."""
        nonlocal assignment_count, backtrack_count
        if all(len(domains[c]) == 1 for c in domains):
            yield capture_state("Solved")
            return True

        cell = pick_cell()
        if cell is None:
            return True if all(len(domains[c]) == 1 for c in domains) else False

        yield capture_state("Select", cell=cell)

        for value in order_values(cell):
            assignment_count += 1
            original_domain = domains[cell]
            domains[cell] = [value]
            yield capture_state("Assign", cell=cell, val=value)

            pruned_domains = {}
            is_valid = True
            if use_forward_checking:
                affected_peers_for_snapshot = []
                for peer_cell in peers[cell]:
                    if value in domains[peer_cell]:
                        if peer_cell not in pruned_domains:
                             pruned_domains[peer_cell] = domains[peer_cell]
                        new_domain = [x for x in domains[peer_cell] if x != value]
                        if not new_domain:
                            is_valid = False
                            domains[peer_cell] = pruned_domains[peer_cell]
                            for p_restore, d_restore in pruned_domains.items():
                                if p_restore != peer_cell:
                                     domains[p_restore] = d_restore
                            pruned_domains = {}
                            break
                        domains[peer_cell] = new_domain
                        affected_peers_for_snapshot.append(peer_cell)
                if is_valid:
                     yield capture_state("Prune", cell=cell, val=value, pruned_cells=affected_peers_for_snapshot)
            if is_valid and (yield from backtrack()):
                return True

            domains[cell] = original_domain
            for peer_cell, original_peer_domain in pruned_domains.items():
                domains[peer_cell] = original_peer_domain

            backtrack_count += 1
            yield capture_state("Backtrack", cell=cell, val=value)

        return False

    yield capture_state("Start")
    if not (yield from backtrack()):
        yield capture_state("Failed")


st.title("🧩 EN.601.464.01: Sudoku Constraint Playground")
st.caption("By: Rafael Chen")

default_state = {
    'steps': [],
    'step_index': 1,
    'initial_grid': None,
    'running': False,
    'delay': 0.75
}
for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar:
    st.header("Puzzle Input & Solver Settings")
    default_puzzle_text = (
        "0,0,3,0,2,0,6,0,0\n"
        "9,0,0,3,0,5,0,0,1\n"
        "0,0,1,8,0,6,4,0,0\n"
        "0,0,8,1,0,2,9,0,0\n"
        "7,0,0,0,0,0,0,0,8\n"
        "0,0,6,7,0,8,2,0,0\n"
        "0,0,2,6,0,9,5,0,0\n"
        "8,0,0,2,0,3,0,0,9\n"
        "0,0,5,0,1,0,3,0,0"
    )
    raw_puzzle_input = st.text_area("Enter 9 rows (comma-separated numbers, 0 for empty):", value=default_puzzle_text, height=250)

    use_fc = st.checkbox("Forward Checking (FC)", True)
    use_mrv = st.checkbox("Minimum Remaining Values (MRV)", True)
    use_lcv = st.checkbox("Least Constraining Value (LCV)", False)

    current_delay = st.session_state.delay
    new_delay = st.slider("Playback Speed (s)", 0.01, 2.0, value=current_delay, step=0.01, key="speed_slider_key")
    if new_delay != current_delay:
        st.session_state.delay = new_delay

    if st.button("Generate Solution Steps"):
        try:
            rows = [r.strip() for r in raw_puzzle_input.splitlines() if r.strip()]
            if len(rows) != 9:
                raise ValueError("Input must contain exactly 9 rows.")
            grid_list = []
            for i, r in enumerate(rows):
                nums = [n.strip() for n in r.split(',')]
                if len(nums) != 9:
                    raise ValueError(f"Row {i+1} ('{r}') must contain exactly 9 numbers.")
                try:
                    grid_list.append(list(map(int, nums)))
                except ValueError:
                     raise ValueError(f"Row {i+1} ('{r}') contains non-numeric values.")
            grid_np = np.array(grid_list)
            if np.any((grid_np < 0) | (grid_np > 9)):
                 raise ValueError("Numbers must be between 0 and 9.")
            st.session_state.initial_grid = grid_np
            st.session_state.steps = list(solve_sudoku_steps(grid_np, use_fc, use_mrv, use_lcv))
            st.session_state.step_index = 1
            st.session_state.running = False
            st.rerun()

        except Exception as e:
            st.sidebar.error(f"Input Error: {e}")

steps_list = st.session_state.steps
total_steps = len(steps_list)
current_index = st.session_state.step_index

if total_steps > 0:
    current_index = max(1, min(current_index, total_steps))
    st.session_state.step_index = current_index
else:
    current_index = 1

if total_steps > 0:
    col_prev, col_play, col_next = st.columns([1, 1, 1])

    if col_prev.button("⬅️ Previous", disabled=(current_index <= 1)):
        st.session_state.running = False
        st.session_state.step_index -= 1
        st.rerun()

    play_pause_placeholder = col_play.empty()
    is_running = st.session_state.running
    if not is_running:
        if play_pause_placeholder.button("▶️ Play", disabled=(current_index >= total_steps)):
            st.session_state.running = True
            st.rerun()
    else:
        if play_pause_placeholder.button("⏸️ Pause", disabled=(current_index >= total_steps)):
            st.session_state.running = False
            st.rerun()
    if col_next.button("Next ➡️", disabled=(current_index >= total_steps)):
        st.session_state.running = False
        st.session_state.step_index += 1
        st.rerun()
    slider_idx = st.slider("Step Navigation", min_value=1, max_value=total_steps, value=current_index, key="step_slider_nav")
    if slider_idx != current_index:
        st.session_state.step_index = slider_idx
        st.session_state.running = False
        st.rerun()

else:
    st.info("Enter a Sudoku puzzle in the sidebar and click 'Generate Solution Steps'.")
    st.stop()

current_state = steps_list[current_index - 1]
grid_to_display = current_state['grid']
reasoning_note = current_state['note']
assignment_count_display = current_state['assignments']
backtrack_count_display = current_state['backtracks']
current_cell_display = current_state['current_cell']
current_action = current_state['action']
affected_cells_display = current_state.get('affected_cells', [])
original_grid_display = st.session_state.initial_grid

grid_col, info_col = st.columns([2, 1])

with info_col:
    st.subheader("🎨 Color Legend")
    legend_html = """
    <ul style="list-style: none; padding-left: 0;">
        <li><span style="display: inline-block; width: 12px; height: 12px; background-color: #fffacd; border: 1px solid #ccc; margin-right: 5px;"></span> Selected Cell</li>
        <li><span style="display: inline-block; width: 12px; height: 12px; background-color: #90ee90; border: 1px solid #ccc; margin-right: 5px;"></span> Assigning Value</li>
        <li><span style="display: inline-block; width: 12px; height: 12px; background-color: #ffcccb; border: 1px solid #ccc; margin-right: 5px;"></span> Backtracking</li>
        <li><span style="display: inline-block; width: 12px; height: 12px; background-color: #add8e6; border: 1px solid #ccc; margin-right: 5px;"></span> Pruned Peer (FC)</li>
    </ul>
    """
    st.markdown(legend_html, unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("🔍 Solver Action")
    st.info(reasoning_note)
    st.subheader("📊 Solver Stats")
    stat_col1, stat_col2 = st.columns(2)
    stat_col1.metric("Assignments", assignment_count_display)
    stat_col2.metric("Backtracks", backtrack_count_display)

    st.subheader("📋 Domain Info")
    if current_cell_display:
        domain_list = current_state['domains'].get(current_cell_display, [])
        st.write(f"Cell {current_cell_display} Domain: `{domain_list}`")
    if current_action == "Prune" and affected_cells_display:
        st.write(f"Pruned {len(affected_cells_display)} peers: `{affected_cells_display}`")


with grid_col:
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xticks(np.arange(9))
    ax.set_yticks(np.arange(9))
    ax.set_xticks(np.arange(-.5, 9, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 9, 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=1)

    for i in range(0, 10, 3):
        ax.axhline(i - 0.5, color='black', linewidth=2)
        ax.axvline(i - 0.5, color='black', linewidth=2)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labelleft=False)

    for r in range(9):
        for c in range(9):
            cell_value = grid_to_display[r, c]
            is_original_number = original_grid_display is not None and original_grid_display[r, c] != 0
            cell_text = str(cell_value) if cell_value else ''
            text_color = 'blue' if is_original_number and cell_value else 'black'
            font_weight = 'bold' if is_original_number and cell_value else 'normal'
            background_color = 'white'
            cell_coord = (r, c)
            if cell_coord == current_cell_display:
                if current_action == "Select": background_color = '#fffacd'
                elif current_action == "Assign": background_color = '#90ee90'
                elif current_action == "Backtrack": background_color = '#ffcccb'
            elif current_action == "Prune" and cell_coord in affected_cells_display:
                 background_color = '#add8e6'
            if background_color != 'white':
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=background_color, alpha=0.6))
            ax.text(c, r, cell_text, ha='center', va='center', fontsize=16, color=text_color, weight=font_weight)

    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(8.5, -0.5)
    st.pyplot(fig, clear_figure=True)
if st.session_state.running and current_index < total_steps:
    time.sleep(st.session_state.delay)
    st.session_state.step_index += 1
    st.rerun()
elif st.session_state.running and current_index >= total_steps:
     st.session_state.running = False
     st.rerun()

st.markdown("---")
st.caption(f"Displaying Step {current_index} of {total_steps}")
