"""
Pathfinding Algorithm Visualizer
================================

This script implements an interactive pathfinding visualizer for the Breadth‑First
Search (BFS), Depth‑First Search (DFS) and Dijkstra algorithms.  It uses
the Pygame library to draw a grid‐based maze and animate the algorithms as
they search for a path between a start and an end cell.  Barriers can be
placed by clicking with the mouse and algorithms can be executed with
keyboard shortcuts.

The project was built to satisfy the requirements of an "Artificial
Intelligence in Action" programming assignment.  A working application,
demonstration video and report are expected deliverables; this module
provides the core of the working application.

Usage
-----

Ensure Python 3.7+ is installed and install Pygame (``pip install pygame``).
Run the script from a terminal:

```
python pathfinding_visualizer.py
```

Controls
--------

* **Left Mouse**: place barriers (black squares)
* **Right Mouse**: remove barriers
* **S Key**: set the start cell (green).  Press once and then click on a cell
  to choose the start.  Only one start cell is allowed; setting a new start
  replaces the previous one.
* **E Key**: set the end cell (red).  Press once and then click on a cell
  to choose the end.  Only one end cell is allowed; setting a new end
  replaces the previous one.
* **B Key**: run Breadth‑First Search
* **D Key**: run Depth‑First Search
* **J Key**: run Dijkstra's algorithm
* **R Key**: reset the grid, clearing barriers but leaving start/end
* **C Key**: clear the grid completely (remove start, end and barriers)

When an algorithm is run the program will animate its progress: visited
nodes turn blue, nodes in the open set turn purple, and the final path
turns yellow.  Once a path is found the algorithm stops and leaves the
grid intact.  Press another algorithm key to see a different search
strategy on the same maze.

Note
----

This script does not require any external resources beyond Pygame.  If
Pygame is not already installed on your system, install it with
``pip install pygame`` prior to execution.  The program creates a
window sized to fit a 50×50 grid by default; adjust the `WIDTH` and
`ROWS` constants below to customize the appearance.
"""

import math
import sys
from collections import deque
from heapq import heappush, heappop
from typing import Callable, Dict, List, Optional, Tuple

try:
    import pygame
except ImportError:
    raise ImportError(
        "Pygame is required to run this visualizer. Install it with 'pip install pygame'."
    )


# -----------------------------------------------------------------------------
# Configuration Constants
#
# Modify these values to change the size of the window or the number of rows.
# The grid will always be square: WIDTH × WIDTH pixels total.  Each cell is
# WIDTH // ROWS pixels wide and tall.  More rows produce smaller cells and a
# finer grid.
WIDTH: int = 600  # size of the window in pixels (square)
ROWS: int = 30    # number of rows/columns in the grid


# -----------------------------------------------------------------------------
# Colour Definitions (R, G, B)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (64, 224, 208)  # turquoise / visited nodes
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)


class Node:
    """Represents a single cell in the grid.

    Nodes know their position (row and column), current colour state and
    neighbouring nodes.  They also expose helper methods for changing state
    (e.g. marking a node as a barrier) and drawing themselves on the screen.
    """

    def __init__(self, row: int, col: int, width: int, total_rows: int) -> None:
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.colour = WHITE
        self.neighbours: List["Node"] = []
        self.width = width
        self.total_rows = total_rows

    # ---------------------------------------------------------------------
    # State query methods

    def is_closed(self) -> bool:
        return self.colour == PURPLE  # nodes in the closed set

    def is_open(self) -> bool:
        return self.colour == BLUE  # nodes discovered but not processed

    def is_barrier(self) -> bool:
        return self.colour == BLACK

    def is_start(self) -> bool:
        return self.colour == GREEN

    def is_end(self) -> bool:
        return self.colour == RED

    # ---------------------------------------------------------------------
    # State update methods

    def reset(self) -> None:
        self.colour = WHITE

    def make_start(self) -> None:
        self.colour = GREEN

    def make_closed(self) -> None:
        # node has been processed (closed set)
        if not self.is_start() and not self.is_end():
            self.colour = PURPLE

    def make_open(self) -> None:
        # node discovered but not yet processed (open set)
        if not self.is_start() and not self.is_end():
            self.colour = BLUE

    def make_barrier(self) -> None:
        self.colour = BLACK

    def make_end(self) -> None:
        self.colour = RED

    def make_path(self) -> None:
        # final path from start to end
        if not self.is_start() and not self.is_end():
            self.colour = YELLOW

    # ---------------------------------------------------------------------
    # Drawing and neighbour management

    def draw(self, win: "pygame.Surface") -> None:
        pygame.draw.rect(win, self.colour, (self.x, self.y, self.width, self.width))

    def update_neighbours(self, grid: List[List["Node"]]) -> None:
        """Update the neighbours list with unblocked adjacent nodes.

        Neighbours are considered only in the four cardinal directions (up,
        down, left and right).  Diagonal movement is not allowed.
        """
        self.neighbours = []
        # Down
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():
            self.neighbours.append(grid[self.row + 1][self.col])
        # Up
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():
            self.neighbours.append(grid[self.row - 1][self.col])
        # Right
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():
            self.neighbours.append(grid[self.row][self.col + 1])
        # Left
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():
            self.neighbours.append(grid[self.row][self.col - 1])

    def __lt__(self, other: "Node") -> bool:
        # Less than operator needed for use in priority queue; doesn't impact
        # equality or hashing, but ensures Python's heapq doesn't crash when
        # comparing Nodes with equal distance values.
        return False


def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Compute the Manhattan distance between two points.

    This heuristic is not used in BFS/DFS/Dijkstra but is provided for
    completeness if further algorithms (e.g. A*) are added.
    """
    (x1, y1), (x2, y2) = a, b
    return abs(x1 - x2) + abs(y1 - y2)


def reconstruct_path(came_from: Dict[Node, Node], current: Node, draw: Callable[[], None]) -> None:
    """Reconstruct the path from start to end after a search completes.

    Starting at the end node, follow the `came_from` dictionary backwards
    through the parents and mark each node as part of the path.  The draw
    function is called between each update to animate the path drawing.
    """
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()


def breadth_first_search(draw: Callable[[], None], grid: List[List[Node]], start: Node, end: Node) -> bool:
    """Perform Breadth‑First Search on the grid from start to end.

    BFS explores neighbouring nodes layer by layer, guaranteeing the
    shortest path in an unweighted graph.  The open set is implemented
    as a queue.  The function returns True if a path was found.
    """
    queue = deque([start])
    came_from: Dict[Node, Node] = {}
    visited = {start}

    while queue:
        current = queue.popleft()

        if current == end:
            # Found the goal; reconstruct and finish
            reconstruct_path(came_from, end, draw)
            end.make_end()
            start.make_start()
            return True

        # Explore neighbours
        for neighbour in current.neighbours:
            if neighbour not in visited and not neighbour.is_barrier():
                visited.add(neighbour)
                came_from[neighbour] = current
                queue.append(neighbour)
                neighbour.make_open()
        draw()
        if current != start:
            current.make_closed()
    return False


def depth_first_search(draw: Callable[[], None], grid: List[List[Node]], start: Node, end: Node) -> bool:
    """Perform Depth‑First Search on the grid from start to end.

    DFS explores as far as possible along each branch before backtracking.
    It does not guarantee the shortest path but may reach the goal quickly
    depending on the maze.  The open set is implemented as a stack.
    """
    stack: List[Node] = [start]
    came_from: Dict[Node, Node] = {}
    visited = {start}

    while stack:
        current = stack.pop()

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            start.make_start()
            return True

        for neighbour in current.neighbours:
            if neighbour not in visited and not neighbour.is_barrier():
                visited.add(neighbour)
                came_from[neighbour] = current
                stack.append(neighbour)
                neighbour.make_open()
        draw()
        if current != start:
            current.make_closed()
    return False


def dijkstra(draw: Callable[[], None], grid: List[List[Node]], start: Node, end: Node) -> bool:
    """Perform Dijkstra's algorithm on the grid from start to end.

    Dijkstra's algorithm computes the shortest path in a graph with non‑negative
    weights.  In this grid the weight of each edge is 1, making it equivalent
    to BFS, but the implementation demonstrates how a priority queue can be
    used to prioritise nodes by cumulative distance.  It also illustrates
    the algorithm's generality for weighted graphs.
    """
    # Distance from start to each node; default to infinity
    distances: Dict[Node, float] = {node: math.inf for row in grid for node in row}
    distances[start] = 0.0
    came_from: Dict[Node, Node] = {}
    visited: set[Node] = set()

    # Priority queue of (distance, node)
    heap: List[Tuple[float, Node]] = []
    heappush(heap, (0.0, start))

    while heap:
        current_distance, current_node = heappop(heap)
        if current_node in visited:
            continue
        visited.add(current_node)

        if current_node == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            start.make_start()
            return True

        for neighbour in current_node.neighbours:
            weight = 1  # weight of moving to an adjacent cell
            tentative_distance = current_distance + weight
            if tentative_distance < distances[neighbour]:
                distances[neighbour] = tentative_distance
                came_from[neighbour] = current_node
                heappush(heap, (tentative_distance, neighbour))
                neighbour.make_open()
        draw()
        if current_node != start:
            current_node.make_closed()
    return False


def make_grid(rows: int, width: int) -> List[List[Node]]:
    """Create a grid of Node objects sized `rows` × `rows`.

    Each node's pixel width is computed by dividing the window width by
    the number of rows.  The grid is returned as a 2‑dimensional list.
    """
    grid: List[List[Node]] = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Node(i, j, gap, rows)
            grid[i].append(node)
    return grid


def draw_grid_lines(win: "pygame.Surface", rows: int, width: int) -> None:
    """Draw grid lines on the window for visual separation of cells."""
    gap = width // rows
    for i in range(rows):
        # Horizontal lines
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        # Vertical lines
        pygame.draw.line(win, GREY, (i * gap, 0), (i * gap, width))


def draw_window(win: "pygame.Surface", grid: List[List[Node]], rows: int, width: int) -> None:
    """Redraw the entire window including grid cells and grid lines."""
    win.fill(WHITE)
    for row in grid:
        for node in row:
            node.draw(win)
    draw_grid_lines(win, rows, width)
    pygame.display.update()


def get_clicked_position(pos: Tuple[int, int], rows: int, width: int) -> Tuple[int, int]:
    """Convert mouse pixel coordinates into (row, column) indices in the grid."""
    gap = width // rows
    x, y = pos
    row = x // gap
    col = y // gap
    return row, col


def reset_grid(grid: List[List[Node]], keep_start_end: bool) -> Tuple[Optional[Node], Optional[Node]]:
    """Reset the grid, optionally preserving the start and end nodes.

    If `keep_start_end` is True the function will remove barriers but
    leave the start and end nodes in place.  It returns a tuple containing
    the (potentially new) start and end nodes.
    """
    start: Optional[Node] = None
    end: Optional[Node] = None
    for row in grid:
        for node in row:
            if node.is_start() and keep_start_end:
                start = node
            elif node.is_end() and keep_start_end:
                end = node
            node.reset()
    if start:
        start.make_start()
    if end:
        end.make_end()
    return start, end


def main(width: int, rows: int) -> None:
    """Main event loop of the program.

    Handles event processing for mouse and keyboard input, updates the grid,
    triggers pathfinding algorithms and redraws the window.  Exits when the
    user closes the window.
    """
    pygame.init()
    win = pygame.display.set_mode((width, width))
    pygame.display.set_caption("Pathfinding Algorithm Visualizer (BFS / DFS / Dijkstra)")

    grid = make_grid(rows, width)
    start: Optional[Node] = None
    end: Optional[Node] = None
    setting_start = False
    setting_end = False

    running = True
    while running:
        draw_window(win, grid, rows, width)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            # Handle mouse clicks
            if pygame.mouse.get_pressed()[0]:  # Left button
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_position(pos, rows, width)
                if row < 0 or row >= rows or col < 0 or col >= rows:
                    continue
                node = grid[row][col]
                if setting_start:
                    # Replace previous start if it exists
                    if start:
                        start.reset()
                    start = node
                    start.make_start()
                    setting_start = False
                elif setting_end:
                    if end:
                        end.reset()
                    end = node
                    end.make_end()
                    setting_end = False
                elif not node.is_start() and not node.is_end():
                    node.make_barrier()

            elif pygame.mouse.get_pressed()[2]:  # Right button
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_position(pos, rows, width)
                if row < 0 or row >= rows or col < 0 or col >= rows:
                    continue
                node = grid[row][col]
                node.reset()
                if node == start:
                    start = None
                elif node == end:
                    end = None

            # Handle key presses
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    # Next left click will set the start node
                    setting_start = True
                    setting_end = False

                elif event.key == pygame.K_e:
                    # Next left click will set the end node
                    setting_end = True
                    setting_start = False

                elif event.key == pygame.K_c:
                    # Completely clear grid
                    start, end = reset_grid(grid, keep_start_end=False)

                elif event.key == pygame.K_r:
                    # Reset barriers but keep start and end
                    start, end = reset_grid(grid, keep_start_end=True)

                # Run BFS
                elif event.key == pygame.K_b and start and end:
                    for row in grid:
                        for node in row:
                            node.update_neighbours(grid)
                    breadth_first_search(lambda: draw_window(win, grid, rows, width), grid, start, end)

                # Run DFS
                elif event.key == pygame.K_d and start and end:
                    for row in grid:
                        for node in row:
                            node.update_neighbours(grid)
                    depth_first_search(lambda: draw_window(win, grid, rows, width), grid, start, end)

                # Run Dijkstra
                elif event.key == pygame.K_j and start and end:
                    for row in grid:
                        for node in row:
                            node.update_neighbours(grid)
                    dijkstra(lambda: draw_window(win, grid, rows, width), grid, start, end)

        # Limit the frame rate to reduce CPU usage
        pygame.time.delay(10)

    pygame.quit()


if __name__ == "__main__":
    # Allow optional command line arguments to override the default grid size.
    # Example: python pathfinding_visualizer.py 800 40
    width = WIDTH
    rows = ROWS
    args = sys.argv[1:]
    if len(args) >= 1:
        try:
            width = int(args[0])
        except ValueError:
            print(f"Invalid width '{args[0]}', using default {WIDTH}")
    if len(args) >= 2:
        try:
            rows = int(args[1])
        except ValueError:
            print(f"Invalid rows '{args[1]}', using default {ROWS}")
    # rows cannot exceed width or be less than 2
    rows = max(2, min(rows, width))
    main(width, rows)