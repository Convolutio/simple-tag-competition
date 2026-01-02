"""
PPO agent for the predator.
"""

from typing import cast
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import numpy as np
from pathlib import Path


class StudentAgent:
    """
    Agent class for Simple Tag competition.
    
    Predator PPO agent logic is implemented here.
    """
    
    def __init__(self):
        """
        Initialize the predator agent.
        """
        # Example: Load your trained models
        # Get the directory where this file is located
        self.submission_dir = Path(__file__).parent
        
        # Example: Load predator model
        # model_path = self.submission_dir / "predator_model.pth"
        # self.model = self.load_model(model_path)
        pass
    
    def get_action(self, observation, agent_id: str):
        """
        Get action for the given observation.
        
        Args:
            observation: Agent's observation from the environment (numpy array)
                         - Predator (adversary): shape (14,)
            agent_id (str): Unique identifier for this agent instance
            
        Returns:
            action: Discrete action in range [0, 4]
                    0 = no action
                    1 = move left
                    2 = move right  
                    3 = move down
                    4 = move up
        """
        # IMPLEMENT YOUR POLICY HERE
        
        # Example random policy (replace with your trained policy):
        # Action space is Discrete(5) by default
        # Note: During evaluation, RNGs are seeded per episode for determinism
        action = np.random.randint(0, 5)
        
        return action
    
    def load_model(self, model_path):
        """
        Helper method to load a PyTorch model.
        
        Args:
            model_path: Path to the .pth file
            
        Returns:
            Loaded model
        """
        # Example implementation:
        # model = YourNeuralNetwork()
        # if model_path.exists():
        #     model.load_state_dict(torch.load(model_path, map_location='cpu'))
        #     model.eval()
        # return model
        pass


class AdversaryTeamAgent(nn.Module):
    """PPO Agent with both the Critic and Actor sub-agents.

    The behavior is the following:
      - the first adversary acts according to a basically learnt behavior
      - the second adversary acts also according to the action and observations
        of the first adversary
      - the others adversaries acts also according the already-taken decisions
        and the already-seen observations

    This is possilbe thanks to contextual information (hidden states) that are
    passed on-the-fly. A RNN layer will be used for that (no need for a LSTM as
    the length of the sequence is the number of adversary agents, so around 3).

    """
    def __init__(self,
                 observation_space_size=14,
                 action_space_size=5,
                 hidden_dim=512):
        super().__init__()
        input_dim = observation_space_size + action_space_size

        # Network
        self.team_observations_encoder = nn.RNN(input_dim, hidden_dim, num_layers=2, nonlinearity="tanh", batch_first=False)
        self.team_decisions_encoder = nn.RNN(input_dim, hidden_dim, num_layers=2, nonlinearity="tanh", batch_first=False)
        # Actor and Critic agents
        self.actor = self._layer_init(nn.Linear(hidden_dim, action_space_size), std=0.01)
        self.critic = self._layer_init(nn.Linear(hidden_dim, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def encode_team_information(self,
                                to: torch.Tensor,
                                ta: torch.Tensor | None=None,
                                to_context: torch.Tensor | None=None,
                                ta_context: torch.Tensor | None=None):
        """
        :param to: Team previous observations. Shape (L, B, observation_space_size)
        :param ta: Team previous actions. Shape (L-1, B, action_space_size) or None if L equals 1
        :param to_context: Encoding of already-seen observations (not in `to` param). Shape (B, hidden_dim)
        :param ta_context: Encoding of already-taken actions (not in `ta` param). Shape (B, hidden_dim)

        :return team_information: Shape (L, B, hidden_dim)
        :return context:
          * new_to_context: The encoded observations. Shape (B, hidden_dim)
          * new_ta_context: The encoded actions. Shape (B, hidden_dim) or None if no action has been provided (meaning no team action has been taken so the first agent is taking a decision)
        """
        to_hidden_states, new_to_context = cast(
            tuple[torch.Tensor, torch.Tensor],
            self.team_observations_encoder(
                to, to_context
            )
        )
        ta_hidden_states, new_ta_context = (
            cast(
                tuple[torch.Tensor, torch.Tensor],
                self.team_decisions_encoder(
                    ta,
                    ta_context)
            )
            if ta is not None else (None, None)
        )
        team_information = (
            to_hidden_states + ta_hidden_states
            if ta_hidden_states is not None
            else to_hidden_states
        )
        return team_information, (new_to_context, new_ta_context)


    def get_value(self,
                  to: torch.Tensor,
                  ta: torch.Tensor | None=None,
                  to_context: torch.Tensor | None=None,
                  ta_context: torch.Tensor | None=None,
                  ):
        hidden, _ = self.encode_team_information(to, ta, to_context, ta_context)
        return self.critic(hidden)

    def get_action_and_value(self,
                             to: torch.Tensor,
                             ta: torch.Tensor | None=None,
                             to_context: torch.Tensor | None=None,
                             ta_context: torch.Tensor | None=None,
                             action=None):
        """
        Arguments:
           see the `encode_team_information` method

        :param action: If provided, force this action to be chosen and return the probs and logits for this action 

        :return action: The chosen action for the adversary agent
        :return context: See the `encode_team_information` method
        :return log_probs: The log probability of the chosen action
        :return entropy: The entropy of the action probabilities
        :return critic: The critic value
        """
        hidden, context = self.encode_team_information(to, ta, to_context,
                                                       ta_context)

        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, context, probs.log_prob(action), probs.entropy(), self.critic(hidden)


if __name__ == "__main__":
    # Example usage
    print("Testing StudentAgent...")
    
    # Test predator agent (adversary has 14-dim observation)
    predator_agent = StudentAgent()
    predator_obs = np.random.randn(14)  # Predator observation size
    predator_action = predator_agent.get_action(predator_obs, "adversary_0")
    print(f"Predator observation shape: {predator_obs.shape}")
    print(f"Predator action: {predator_action} (should be in [0, 4])")
    
    print("✓ Agent template is working!")
