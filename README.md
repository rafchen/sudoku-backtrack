# Sudoku Constraint Playground

This playground provides an interactive visualization of the backtracking algorithm used to solve Sudoku puzzles. It allows users to observe the step-by-step process and understand the impact of different constraint satisfaction heuristics.

Try it yourself on : https://rafchen-sudoku-backtrack-app-kmv0ra.streamlit.app/

## Features

*   **Interactive Sudoku Grid:** Visualize the puzzle being solved in real-time.
*   **Step-by-Step Playback:** Control the visualization speed, pause, step forward/backward through the solving process.
*   **Heuristic Selection:** Enable or disable common constraint satisfaction heuristics:
    *   **Forward Checking (FC):** Prunes the domains of neighboring cells after assigning a value.
    *   **Minimum Remaining Values (MRV):** Selects the next cell to assign based on the one with the fewest legal values remaining.
    *   **Least Constraining Value (LCV):** Orders the potential values for a cell based on which one rules out the fewest choices for neighboring cells.
*   **Color-Coded Visualization:** Different colors highlight:
    *   The currently selected cell.
    *   Cells being assigned a value.
    *   Cells involved in backtracking.
    *   Peer cells whose domains are pruned by Forward Checking.
*   **Solver Statistics:** Track the number of assignments and backtracks performed.
*   **Domain Information:** View the current domain (possible values) for the selected cell.
*   **Custom Puzzle Input:** Enter your own Sudoku puzzles to solve and visualize.

## Installation

1.  Clone the repository (if you haven't already).
2.  Install required Python libraries:
    pip install streamlit numpy matplotlib

## Usage

1.  Navigate to the directory containing `app.py` in your terminal:
    cd /path/to/sudoku-backtrack
2.  Run the application:
    streamlit run app.py
3.  The application will open in your web browser.
4.  Use the sidebar to:
    *   Enter the Sudoku puzzle (0 represents empty cells).
    *   Select the desired heuristics (FC, MRV, LCV).
    *   Adjust the playback speed.
    *   Click "Generate Solution Steps".
5.  Use the playback controls (Previous, Play/Pause, Next) and the slider to navigate through the solving steps.
6.  Observe the grid visualization and the information panel for details about each step.
