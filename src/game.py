"""G-Game (Grounded Game) implementation based on Visser's thesis.

This module implements the grounded argument game following Visser's approach,
adapted for Python and integrated with the existing ArgumentationFramework.

Key aspects of the G-game:
1. Players alternate: PRO, CON, PRO, CON, ...
2. Roles alternate: P, O, P, O, ... (each move gets opposite role from target)
3. Initial move: PRO with role P, target_move_id = 0
4. Rule: Player with role P cannot repeat arguments in same line of dispute
5. Any player repeating their own argument in same line ends that line
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from argumentation import ArgumentWrapper


class Player(Enum):
    """Players in the grounded game."""

    PRO = "PRO"
    CON = "CON"

    def get_other_player(self) -> Player:
        """Get the opposing player."""
        return Player.CON if self == Player.PRO else Player.PRO


class Role(Enum):
    """Roles in the grounded game (Proponent/Opponent)."""

    P = "P"
    O = "O"

    def get_other_role(self) -> Role:
        """Get the opposing role."""
        return Role.O if self == Role.P else Role.P


class AttackType(Enum):
    """Types of attacks in the grounded game."""

    DEFEAT = "DEFEAT"


class GameOutcome(Enum):
    """Possible outcomes of the grounded game."""

    PRO_WINS = "PRO_WINS"
    CON_WINS = "CON_WINS"
    UNDECIDED = "UNDECIDED"


@dataclass
class Move:
    """Represents a single move in the grounded game."""

    move_id: int
    player: Player
    role: Role
    argument: ArgumentWrapper
    target_move_id: int
    target_argument: ArgumentWrapper | None
    attack_type: AttackType
    is_backtracked: bool = False

    def backtrack(self) -> None:
        """Mark this move as backtracked."""
        self.is_backtracked = True

    def get_description(self) -> str:
        """Get human-readable description of the move."""
        backtrack_str = " [BACKTRACKED]" if self.is_backtracked else ""

        if self.target_move_id == 0:
            return (
                f"Move {self.move_id}: {self.player.value},{self.role.value} "
                f"puts forward {self.argument.name}{backtrack_str}"
            )
        else:
            target_name = (
                self.target_argument.name if self.target_argument else "unknown"
            )
            return (
                f"Move {self.move_id}: {self.player.value},{self.role.value} "
                f"{self.attack_type.value.lower()}s {target_name} with {self.argument.name} "
                f"(targeting move {self.target_move_id}){backtrack_str}"
            )


class GroundedGame:
    """Implements the grounded argument game (G-game) following Visser's approach."""

    def __init__(self, framework: Any, predicted_outcome, focus_case):
        """Initialise the grounded game."""
        self.framework = framework
        self.predicted_outcome = predicted_outcome
        self.game: dict[int, Move] = {}
        self.move_id_counter = 1
        self.focus_case = focus_case

    def play_game(self, best_precedents) -> GameOutcome:
        """Play the grounded game to determine if query argument is justified."""
        from af_cba import Citation

        citation_arguments = []
        for idx, precedent in best_precedents.iterrows():

            precedent_name = (
                str(idx) if not hasattr(precedent, "name") else str(precedent.name)
            )

            citation = Citation(
                case=precedent,
                focus_case=self.focus_case,
                case_name=precedent_name,
                introduced_by="PRO",
                responding_to=None,
            )

            if hasattr(self.framework, "af"):
                self.framework.af.add_argument(citation.name, citation.content)
                self.framework._generated_arguments.add(citation)
            else:
                self.framework.add_argument(citation.name, citation.content)

            citation_arguments.append(citation)

        self._play_recursive(Player.PRO, citation_arguments, 0)

        return self._determine_outcome()

    def _play_recursive(
        self,
        player: Player,
        possible_arguments: list[ArgumentWrapper],
        target_move_id: int,
    ) -> None:
        """Recursively construct the game tree following Visser's G-game rules."""
        if not possible_arguments:
            # Before giving up, check if we can generate more counterexamples
            if target_move_id > 0:
                target_argument = self.game[target_move_id].argument
                next_counterexample = self.lazy_framework.get_next_counterexample(
                    target_argument
                )
                if next_counterexample:
                    possible_arguments = [next_counterexample]
                    print(f"Generated new counterexample: {next_counterexample.name}")

        if not possible_arguments:
            # No moves available, player backtracks
            return

        for argument in possible_arguments:

            if self._is_last_player(player):
                return

            if player == Player.PRO:
                role = Role.P
            else:
                role = Role.O

            if not self._is_legal_move(player, argument, target_move_id, role):
                continue

            this_move_id = self.move_id_counter
            if role == Role.P and this_move_id != target_move_id + 1:

                self._backtrack_moves(target_move_id + 1, this_move_id)

            current_move = Move(
                move_id=this_move_id,
                player=player,
                role=role,
                argument=argument,
                target_move_id=target_move_id,
                target_argument=(
                    self.game[target_move_id].argument if target_move_id > 0 else None
                ),
                attack_type=AttackType.DEFEAT,
            )

            self.game[this_move_id] = current_move

            self.move_id_counter += 1

            if self._played_before_in_line(argument, target_move_id, player):

                return

            replies = self._get_replies_for_argument(argument)

            other_player = player.get_other_player()
            self._play_recursive(other_player, replies, current_move.move_id)

    def _is_citation_argument(self, argument: ArgumentWrapper) -> bool:
        """Check if an argument is a Citation argument from AF-CBA."""
        from af_cba import Citation

        if isinstance(argument, Citation):
            return True

        if argument.content:
            try:
                import json

                content_data = json.loads(argument.content)
                return content_data.get("type") == "Citation"
            except (json.JSONDecodeError, KeyError):
                pass

        return argument.name.startswith("Citation(")

    def _is_last_player(self, player: Player) -> bool:
        """Check if player was the last to move in the game."""
        if not self.game:
            return False

        last_move_id = max(self.game.keys())
        last_move = self.game[last_move_id]
        return last_move.player == player

    def _is_legal_move(
        self, player: Player, argument: ArgumentWrapper, target_move_id: int, role: Role
    ) -> bool:
        """Check if a move is legal under G-game rules."""
        if role == Role.P and self._played_before_in_line(
            argument, target_move_id, None
        ):
            return False

        return True

    def _played_before_in_line(
        self, argument: ArgumentWrapper, target_move_id: int, player: Player | None
    ) -> bool:
        """Check if argument was played before in this line of dispute."""
        if target_move_id == 0:
            return False

        current_move = self.game[target_move_id]

        if current_move.argument == argument:
            if player is None or current_move.player == player:
                return True

        return self._played_before_in_line(
            argument, current_move.target_move_id, player
        )

    def _backtrack_moves(self, start_move_id: int, end_move_id: int) -> None:
        """Mark moves as backtracked."""
        for move_id in range(start_move_id, end_move_id):
            if move_id in self.game:
                self.game[move_id].backtrack()

    def _get_replies_for_argument(
        self, argument: ArgumentWrapper
    ) -> list[ArgumentWrapper]:
        """Get all arguments that can attack the given argument."""
        attackers = self.framework.get_attackers(argument.name)
        return list(attackers)

    def _determine_outcome(self) -> GameOutcome:
        """Determine the outcome based on the constructed game tree."""
        if not self.game:
            return GameOutcome.UNDECIDED

        last_player = None
        for move_id in sorted(self.game.keys(), reverse=True):
            move = self.game[move_id]
            if not move.is_backtracked:
                last_player = move.player
                break

        if last_player == Player.PRO:
            return GameOutcome.PRO_WINS
        elif last_player == Player.CON:
            return GameOutcome.CON_WINS
        else:
            raise RuntimeError("Undecided")
            return GameOutcome.UNDECIDED

    def is_query_justified(self) -> bool:
        """Determine if the query argument is justified."""
        outcome = self.play_game() if not self.game else self._determine_outcome()
        return outcome == GameOutcome.PRO_WINS

    def get_game_tree_description(self) -> str:
        """Get a human-readable description of the game tree."""
        if not self.game:
            return "Empty game"

        lines = ["Grounded Game Tree:"]
        lines.append(f"Query: {self.predicted_outcome}")
        lines.append("")

        for move_id in sorted(self.game.keys()):
            move = self.game[move_id]
            indent = "  " * self._get_move_depth(move_id)
            lines.append(f"{indent}{move.get_description()}")

        return "\n".join(lines)

    def _get_move_depth(self, move_id: int) -> int:
        """Calculate the depth of a move in the game tree."""
        if move_id not in self.game:
            return 0

        move = self.game[move_id]
        if move.target_move_id == 0:
            return 0
        else:
            return 1 + self._get_move_depth(move.target_move_id)

    def get_winning_strategy(
        self, visualisation_path: str | None = None
    ) -> list[Move] | None:
        """Extract the winning strategy from the game tree with optional visualisation."""
        if not self.game:
            return None

        main_line = []
        for move_id in sorted(self.game.keys()):
            move = self.game[move_id]
            if not move.is_backtracked:
                main_line.append(move)

        if visualisation_path and main_line:
            self._create_winning_strategy_visualisation(visualisation_path, main_line)

        return main_line if main_line else None

    def _create_winning_strategy_visualisation(
        self, output_path: str, winning_strategy: list[Move]
    ) -> None:
        """Create a visualisation highlighting the winning strategy arguments."""
        try:
            from visualisation import ArgumentationFrameworkVisualiser

            strategy_argument_names = {move.argument.name for move in winning_strategy}

            visualiser = ArgumentationFrameworkVisualiser()

            outcome = self._determine_outcome()
            title = f"Grounded Game: {self.query_argument.name} - {outcome.value}"

            visualiser.create_graph_visualisation(
                framework=self.framework,
                output_path=output_path,
                title=title,
                highlight_grounded=True,
                highlight_strategy=strategy_argument_names,
                layout="hierarchical",
            )

        except ImportError:

            pass
