# Particle Life

A particle life simulation built with Python, NumPy, and Pygame. Particles of different types interact with each other through attraction and repulsion forces, producing complex emergent behaviour from simple rules.

## Demo

Particles are rendered as coloured dots on a black background with toroidal wrapping (particles that leave one edge re-enter from the opposite side).

## How It Works

Each particle has a type (one of 6 colours). An **attraction matrix** defines how strongly each type is attracted to or repelled by every other type. At each timestep:

1. A **spatial hash** partitions the screen into a grid of cells to avoid checking every pair of particles (reduces O(N²) to roughly O(N)).
2. For each particle, only neighbours in adjacent grid cells are considered.
3. Forces are calculated using a piecewise linear model:
   - **Too close** (`dist < R_MIN`): repulsion
   - **In range** (`R_MIN ≤ dist < R_MAX`): attraction or repulsion based on the matrix
   - **Far away** (`dist ≥ R_MAX`): no effect
4. Velocities are updated and damped by a friction factor.
5. Positions are updated and wrapped to the screen bounds.

The physics loop is JIT-compiled and parallelised with [Numba](https://numba.readthedocs.io/) for performance.

## Configuration

All simulation parameters are defined at the top of `code/app.py`:

| Parameter | Default | Description |
|---|---|---|
| `WIDTH`, `HEIGHT` | 1920 × 1200 | Window resolution |
| `N_PARTICLES` | 7500 | Total number of particles |
| `N_TYPES` | 6 | Number of particle types |
| `R_MIN` | 50 | Minimum interaction radius (repulsion zone) |
| `R_MAX` | 200 | Maximum interaction radius |
| `FRICTION` | 0.9 | Velocity damping per frame |
| `FPS` | 60 | Target frame rate |
| `DT` | 0.01 | Time step scalar |

The attraction matrix is randomly generated on each run (seeded with `np.random.seed(25)` for reproducibility). Each particle type is always self-attracting — the diagonal is forced positive.

## Requirements

- Python 3.13
- pygame
- numpy
- numba

Install dependencies into the included virtual environment or your own:

```bash
pip install pygame numpy numba
```

## Running

```bash
python code/app.py
```

Press `Escape` or close the window to exit.

## Controls

| Key | Action |
|---|---|
| `Escape` | Quit |
| Window close button | Quit |

## Project Structure

```
particle life/
├── code/
│   └── app.py        # Main simulation
├── basic.py          # Earlier prototype
└── particle_life/    # Python virtual environment
```

## Implementation Notes

Three velocity update implementations are present in the code:

- `update_velocities` — simple N² loop using NumPy vector operations
- `update_velocities_components` — N² loop broken into scalar components for better cache usage
- `update_velocities_components_hash` — spatial hash accelerated version (active)

The spatial hash divides the screen into `R_MAX × R_MAX` sized cells. For each particle, only the 9 surrounding cells are checked, making the simulation practical at 7500+ particles.
