""" PPO agent for the predator. """
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
    
    def __init__(self, training=False):
        """
        Initialize the predator agent.
        """
        # Example: Load your trained models
        # Get the directory where this file is located
        self.submission_dir = Path(__file__).parent
        
        # Example: Load predator model
        model_path = self.submission_dir / "predator_model.pth"
        self.model = self.load_model(model_path, training)
    
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
        action, _, _, _ = self.model.forward_for_agent(observation,
                                                          agent_id)
        
        return action
    
    def load_model(self, model_path: Path, training=False):
        """
        Helper method to load a PyTorch model.
        
        Args:
            model_path: Path to the .pth file
            
        Returns:
            Loaded model
        """
        model = get_global_team_agent()
        if model is not None:
            return model
        model = AdversaryTeamAgent()
        if not training and model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
        set_global_team_agent(model)
        return model


class AdversaryTeamAgent(nn.Module):
    """PPO Agent with both the Critic and Actor sub-agents.

    The behavior is the following:
      - the first adversary acts according to its observations
      - the second adversary acts also according to the action of the first
        adversary
      - the others adversaries acts also according the already-taken decisions

    This is possilbe thanks to contextual information (hidden states) that are
    passed on-the-fly. A RNN layer will be used for that (no need for a LSTM as
    the length of the sequence is the number of adversary agents, so around 3).

    """
    def __init__(self,
                 observation_feat_nb=14,
                 action_number=5,
                 hidden_dim=512,
                 action_embedding_dim=2,
                 ):
        super().__init__()
        input_dim = observation_feat_nb

        # Network
        self.agent_observation_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.team_decisions_encoder = nn.Sequential(
            nn.Embedding(action_number, action_embedding_dim),
            nn.RNN(
                action_embedding_dim, hidden_dim, num_layers=2,
                nonlinearity="tanh", batch_first=False
            )
        )
        # Actor and Critic agents
        self.actor = self._layer_init(nn.Linear(hidden_dim, action_number), std=0.01)
        self.critic = self._layer_init(nn.Linear(hidden_dim, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer


    # --- ENCODING LAYER ---


    def encode_on_the_fly_team_information(
        self,
        unit_observations: torch.Tensor,
        previous_action: torch.Tensor | None=None,
        prior_previous_action_ctx: torch.Tensor | None=None
    ):
        """Return the hidden values of the network, with contextual information.

        Expect to encode the observation of one unit (evaluation/forward
        stage), with the action of the previous unit (if the current unit is
        not the first of the team) and the encoded context of chosen actions of
        the prior units (if the current unit is at least the 3rd unit of the
        team).

        :param unit_observations: shape (B, observation_space_size)
        :param previous_action: shape (B,)
        :param prior_previous_action_ctx: shape (B, hidden_dim)

        :return hidden_features: shape (B, hidden_dim)
        :return previous_action_ctx: shape (B, hidden_dim)
        """
        observation_hidden_features = cast(
            torch.Tensor,
            self.agent_observation_encoder(unit_observations)
        )  # shape (B, hidden_dim)
        (
            _,
            previous_action_ctx  # shape (B, hidden_dim)
        ) = (
            cast(
                tuple[torch.Tensor, torch.Tensor],
                self.team_decisions_encoder(
                    previous_action.unsqueeze(0),
                    prior_previous_action_ctx
                )
            )
            if previous_action is not None else (None, None)
        )
        hidden_features = (
            # combine the observation features with the team action information
            # when existing
            observation_hidden_features + previous_action_ctx
            if previous_action_ctx is not None
            else observation_hidden_features
        )
        return hidden_features, previous_action_ctx


    def encode_team_information(
        self,
        team_observations: torch.Tensor,
        chosen_actions: torch.Tensor
    ):
        """Return the hidden values of the network, with contextual information.

        Expect to encode the observations of all the team (training stage). See
        the `encode_on_the_fly_team_information` method for an on-the-fly usage
        suitable for the evaluation/forward stage.

        Let L be the number of units in the team.
        The L-1 actions will be used to compute the team contextual information
        for all the units.

        :param unit_observations: Current agent's observations. Shape (L, B, observation_space_size), with L the number of units in the team (when several units are passed in the same batch)
        :param chosen_actions: Team chosen actions. Shape (L, B)

        :return team_information: Shape (L, B, hidden_dim)
        """
        observation_hidden_features = self.agent_observation_encoder(
            team_observations
        )  # shape (L, B, hidden_dim)
        team_action_hidden_states__partial, _ = cast(
            tuple[torch.Tensor, torch.Tensor],
            self.team_decisions_encoder(
                chosen_actions[:-1],
            )
        )  # shape (L-1, B, hidden_dim)
        team_action_hidden_states = torch.cat(
            (
                # the first unit does not have any contextual information
                # the detached zeros are neutral in the final encoding addition
                torch.zeros_like(team_action_hidden_states__partial[0]).detach()
                .to(team_action_hidden_states__partial.device),
                team_action_hidden_states__partial
            ), dim=0,
        )  # shape (L, B, hidden_dim)
        # combine the observation features with the team action information
        return observation_hidden_features + team_action_hidden_states


    # --- ACTOR CRITIC HEADS ---


    def actor_critic_forward(self, hidden: torch.Tensor, action_s: torch.Tensor | None):
        """Return the results of the actor and critic heads.

        This function works for decisions for both one unit or the full team.

        :param hidden: shape (B, hidden_dim) or (L, B, hidden_dim)
        :param action_s: shape (B,) or (L, B)

        TODO: check the shapes

        :return action_s: The chosen action(s) for the adversary agent(s). Shape (B,) or (L, B)
        :return log_probs: The log probability(ies) of the chosen action(s). Shape (B,) or (L, B)
        :return entropy: The entropy of the action probabilities. Shape (B, action_number) or (L, B, action_number)
        :return critic: The critic value. Shape (B,) or (L, B)
        """
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action_s is None:
            action_s = probs.sample()
        return action_s, probs.log_prob(action_s), probs.entropy(), self.critic(hidden)


    # --- FORWARD ---


    def get_action_and_value(self,
                             team_observations: torch.Tensor,
                             actions: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor
    ]:
        """
        Arguments:
           see the `encode_team_information` method

        :param action: If provided, force the actions to be chosen and return the probs and logits for these actions.

        Return:
           see the `actor_critic_forward` method.
        """

        hidden = self.encode_team_information(team_observations, actions)
        return self.actor_critic_forward(hidden, actions)


    def get_on_the_fly_action_and_value(
        self,
        unit_observations: torch.Tensor,
        previous_action: torch.Tensor | None=None,
        prior_previous_action_ctx: torch.Tensor | None=None
    ):
        """
        Arguments:
           see the `encode_on_the_fly_team_information` method

        :return action_s: The chosen action for the adversary agent. Shape (B,)
        :return context: The context features of the team previous action. Shape (B, hidden_dim)
        :return log_probs: The log probability of the chosen action. Shape (B,)
        :return entropy: The entropy of the action probabilities. Shape (B, action_number)
        :return critic: The critic value. Shape (B,)
        """
        hidden, context = self.encode_on_the_fly_team_information(
            unit_observations,
            previous_action,
            prior_previous_action_ctx
        )
        action, log_probs, entropy, critic = self.actor_critic_forward(hidden, None)
        return action, context, log_probs, entropy, critic


    # --- TEAM Actions ---
    def start_team_step(self):
        self._first_seen_agent_in_team: str | None = None
        self._last_teammate_action: torch.Tensor | None = None
        self._team_action_ctx: torch.Tensor | None = None

    def forward_for_agent(self, agent_observation: torch.Tensor, agent_id: str):
        if (
            self._first_seen_agent_in_team is None
            or self._first_seen_agent_in_team == agent_id
        ):
            # a new step for the whole team starts
            self.start_team_step()
            self._first_seen_agent_in_team = agent_id
        # return the actions for the current adversary
        new_action, new_action_ctx, log_probs, entropy, critic_value = (
            self.get_on_the_fly_action_and_value(
                agent_observation,
                self._last_teammate_action,
                self._team_action_ctx,
            )
        )
        # update the context for the next adversaries
        self._last_teammate_action = new_action
        self._team_action_ctx = new_action_ctx
        return new_action, log_probs, entropy, critic_value


    def get_on_the_fly_action_and_value_for_team(
        self,
        team_observations: torch.Tensor,
        team_agent_ids: list[str]
    ):
        """

        :param team_observations: shape (L, B, observation_feat_nb)
        :param team_agent_ids: list of L elements
        """
        team_size, b, _ = team_observations.shape
        actions = torch.empty((team_size, b)).to(team_observations.device)
        log_probs = torch.empty((team_size, b)).to(team_observations.device) 
        entropy = torch.empty((team_size, b, 5))  # TODO: rename in action_nb
        critic_values = torch.empty((team_size, b)).to(team_observations.device)

        # TODO: zip and concatenate
        for k, agent_id in enumerate(team_agent_ids):
            action, lp, entr, crit = self.forward_for_agent(team_observations[k], agent_id)
            actions[k] = action
            log_probs[k] = lp
            entropy[k] = entropy


__global_team_agent: AdversaryTeamAgent | None = None

def get_global_team_agent() -> AdversaryTeamAgent | None:
    """Init the team agent or only return it, if already initiated."""
    global __global_team_agent
    return __global_team_agent

def set_global_team_agent(team_agent: AdversaryTeamAgent):
    global __global_team_agent
    __global_team_agent = team_agent



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
