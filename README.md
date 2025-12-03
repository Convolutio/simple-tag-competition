# Simple Tag Competition - RL Class Project

Welcome to the Simple Tag Competition! This project uses the [Simple Tag environment from PettingZoo's MPE](https://pettingzoo.farama.org/environments/mpe/simple_tag/).

## Overview

In this competition, you will train the **predator** agent only. Your predator will be evaluated against a public reference **prey** provided in this repository.

## Environment

- **Simple Tag**: A cooperative/competitive multi-agent environment where three predators try to catch a prey
- Three predators chase a single prey in a continuous 2D space
- Predators are rewarded for catching prey; the prey is rewarded for avoiding capture

## Submission Guidelines

### What to Submit

You must submit **via Pull Request** from your fork:

1. **One Python file**: `agent.py` - Contains your agent implementation
2. **Model weights** (optional): Any `.pth` files for your neural networks

### File Structure

Your submission should follow this structure:
```
submissions/<your_username>/
├── agent.py           # Required: Your predator implementation
└── predator_model.pth # Optional: Your predator neural network weights
```

### Agent Implementation Requirements

Your `agent.py` must implement the following class (predator only):

```python
class StudentAgent:
    def __init__(self):
        """
        Initialize your predator agent.
        """
        pass
    
    def get_action(self, observation, agent_id: str):
        """
        Get action for the given observation.
        
        Args:
            observation: Agent's observation from the environment
            agent_id: Unique identifier for this agent instance
            
        Returns:
            action: Action to take in the environment
        """
        pass
```

See `template/agent.py` for a complete template.

### How to Submit

1. **Fork** this repository
2. **Create** your submission folder: `submissions/<your_username>/`
3. **Add** your `agent.py` (and optional `.pth` files)
4. **Create a Pull Request** to the main repository
5. **Wait** for automatic evaluation - results will appear on the [leaderboard](https://nathanael-fijalkow.github.io/simple-tag-competition/)

### Evaluation

- Your predator is evaluated against a **public reference prey**
- Each PR triggers automatic evaluation via GitHub Actions
- Results are published to the leaderboard immediately
- **Note**: PRs are not merged - they are only used for evaluation

### Rules

- You may use any RL algorithm (DQN, PPO, SAC, etc.)
- You may train your agents however you like
- You may use pre-trained models
- You can submit multiple times (new PRs will update your score)
- Do not modify files outside your submission folder
- Submit one `agent.py` file and optionally any `.pth` model files

## Local Development

### Installation

```bash
# Clone the repository
git clone https://github.com/nathanael-fijalkow/simple-tag-competition.git
cd simple-tag-competition

# Install dependencies
pip install -r requirements.txt
```

### Testing Your Agent Locally

```bash
python test_agent.py submissions/<your_username>/agent.py
```

## Leaderboard

Check the live leaderboard at: [https://nathanael-fijalkow.github.io/simple-tag-competition/](https://nathanael-fijalkow.github.io/simple-tag-competition/)

The leaderboard shows:
- Student username
- Predator score (average reward vs public prey)
- Submission timestamp
- Ranking
