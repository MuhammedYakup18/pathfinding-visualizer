# Pathfinding Algorithm Visualizer

An interactive graphical tool built in Python using Pygame to visualize popular pathfinding algorithms. This project demonstrates how different search strategies navigate a grid-based maze to find a path between a start and an end point.

## Supported Algorithms

* **Breadth-First Search (BFS)**: Explores layer by layer. Guarantees the shortest path in an unweighted grid.
* **Depth-First Search (DFS)**: Explores as far as possible along a branch before backtracking. Does not guarantee the shortest path.
* **Dijkstra's Algorithm**: Prioritizes nodes by cumulative distance. While equivalent to BFS on this unweighted grid, the implementation demonstrates the use of a priority queue, making it extensible for weighted graphs.

## Requirements

* Python 3.7+
* [Pygame](https://www.pygame.org/)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/MuhammedYakup18/pathfinding-visualizer.git
   cd pathfinding-visualizer
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the script from your terminal:

```bash
python pathfinding_visualizer.py
```

You can optionally specify the window width and the number of rows/columns:
```bash
python pathfinding_visualizer.py [window_width] [number_of_rows]
# Example: python pathfinding_visualizer.py 800 40
```

## Controls

* **Left Mouse Click**: Place barriers (black squares).
* **Right Mouse Click**: Remove barriers.
* **S Key**: Set the start node (green). Press `S` once, then left-click on a cell.
* **E Key**: Set the end node (red). Press `E` once, then left-click on a cell.
* **B Key**: Run Breadth-First Search.
* **D Key**: Run Depth-First Search.
* **J Key**: Run Dijkstra's Algorithm.
* **R Key**: Reset the grid (clears barriers and paths but keeps the start/end points).
* **C Key**: Clear the grid completely (removes everything).

## Color Legend

* **White**: Empty cell
* **Black**: Barrier
* **Green**: Start node
* **Red**: End node
* **Blue**: Open set (nodes discovered but not yet processed)
* **Purple**: Closed set (nodes that have been processed)
* **Yellow**: Final path
