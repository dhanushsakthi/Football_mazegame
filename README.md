# Football_maze-game

# Football Maze Game 🏈⚽

An interactive maze-solving game where a football navigates through a field to reach the goal post using AI pathfinding algorithms!

## 🎮 Game Overview

This is a Python-based maze game that combines reinforcement learning (Q-learning) with traditional pathfinding algorithms (A*) to create an intelligent football agent. The player clicks to set start and goal positions, and watches as the AI finds the optimal path while the football smoothly rolls across the field.

## ✨ Features

- **Interactive Maze Generation**: Randomly generated maze layouts
- **Dual AI Approach**: 
  - Primary: Q-learning reinforcement learning agent
  - Fallback: A* pathfinding algorithm
- **Smooth Animations**: Rolling football with realistic movement
- **Victory Celebration**: Animated celebration sequence when goal is reached
- **Football Theme**: Grass field, football sprite, and goal post graphics
- **Real-time Training Visualization**: Watch the AI learn in real-time

## 🛠️ Requirements

### Dependencies
```bash
pip install pygame numpy pillow
```

### Required Assets
Make sure these files are in the same directory as the Python script:

- `grass_field.png` - Background grass texture
- `football.png` - Static football image
- `goal_post.png` - Goal post image
- `football.gif` - Animated rolling football
- `celebration.gif` - Victory celebration animation

## 🚀 How to Run

1. Clone this repository
2. Install the required dependencies
3. Ensure all asset files are in the project directory
4. Run the game:
```bash
python full_time.py
```

## 🎯 How to Play

1. **Launch the game** - A 10x10 maze grid will appear
2. **Set start position** - Click anywhere on the grid to place the football
3. **Set goal position** - Click a second location to place the goal post
4. **Watch the magic** - The AI will:
   - Train using Q-learning (8 episodes)
   - Find the optimal path
   - Animate the football rolling to the goal
   - Show a victory celebration!

## 🧠 AI Algorithms

### Q-Learning (Primary)
- **Alpha (Learning Rate)**: 0.1
- **Gamma (Discount Factor)**: 0.9
- **Epsilon (Exploration Rate)**: 0.2
- **Training Episodes**: 8
- **Reward System**:
  - Goal reached: +100
  - Wall collision: -50
  - Each step: -1

### A* Pathfinding (Fallback)
- Used when Q-learning fails or goal is unreachable
- Manhattan distance heuristic
- Guarantees optimal path if one exists

## 🏗️ Code Structure

```
full_time.py
├── Maze Generation (generate_maze)
├── Environment Class (MazeEnv)
├── Pathfinding Algorithms
│   ├── BFS Reachability Check
│   ├── A* Implementation
│   └── Q-Learning Training
├── Graphics & Animation
│   ├── GIF Frame Loader
│   ├── Smooth Path Animation
│   └── Victory Celebration
└── Main Game Loop
```

## ⚙️ Configuration

You can modify these constants at the top of the file:

```python
WIDTH, HEIGHT = 600, 600    # Window size
ROWS, COLS = 10, 10         # Maze dimensions
alpha = 0.1                 # Learning rate
gamma = 0.9                 # Discount factor
epsilon = 0.2               # Exploration rate
```

## 🎨 Customization

### Adding New Themes
1. Replace the asset files with your own images/GIFs
2. Ensure they match the expected filenames
3. Adjust `CELL_SIZE` if needed for different maze dimensions

### Modifying AI Behavior
- Increase training episodes for better learning
- Adjust reward values in the `step()` method
- Modify exploration rate (epsilon) for different behaviors

## 🐛 Troubleshooting

**Game won't start:**
- Check that all required image/GIF files exist
- Verify pygame installation: `pip install pygame`

**AI gets stuck:**
- The A* fallback should handle most cases
- Try clicking different start/goal positions

**Poor performance:**
- Reduce maze size (ROWS, COLS)
- Decrease animation frames
- Lower training episodes

## 🤝 Contributing

Feel free to fork this project and submit pull requests! Some ideas for improvements:

- Multiple difficulty levels
- Different maze generation algorithms
- More sophisticated AI agents
- Sound effects and music
- High score system
- Multiple football themes

## 📝 License

This project is open source. Feel free to use, modify, and distribute as needed.

## 🙏 Acknowledgments

- Built with Python, Pygame, NumPy, and Pillow
- Inspired by classic maze-solving algorithms and modern reinforcement learning
- Perfect for learning AI concepts in a fun, visual way!

---

**Enjoy watching your AI football player navigate the maze! 🏈🎯**
