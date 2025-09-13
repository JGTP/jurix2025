"""Visualisation module for AF-CBA winning strategies.

Provides clean, focused visualisations of argumentation frameworks showing
only the winning strategy arguments and their defeat relations.
"""

import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from argumentation import ArgumentationFramework


def create_winning_strategy_framework(
    full_framework: ArgumentationFramework, strategy_argument_names: set[str]
) -> ArgumentationFramework:
    """Create a new framework containing only the winning strategy arguments."""
    strategy_framework = ArgumentationFramework(
        f"{full_framework.name}_winning_strategy"
    )

    for arg in full_framework.get_arguments():
        if arg.name in strategy_argument_names:
            strategy_framework.add_argument(arg.name, arg.content)

    for defeat in full_framework.get_defeats():
        if (
            defeat.from_arg.name in strategy_argument_names
            and defeat.to_arg.name in strategy_argument_names
        ):
            strategy_framework.add_defeat(defeat.from_arg.name, defeat.to_arg.name)

    return strategy_framework


class ArgumentationFrameworkVisualiser:
    """Visualiser for ArgumentationFramework objects focused on winning strategies."""

    def __init__(self, figsize: tuple[float, float] = (12, 8)):
        """Initialise the visualiser."""
        self.figsize = figsize

    def create_winning_strategy_visualisation(
        self,
        full_framework: ArgumentationFramework,
        strategy_argument_names: set[str],
        output_path: str,
        title: str | None = None,
        layout: str = "hierarchical",
    ) -> None:
        """Create a clean visualisation showing only the winning strategy."""
        strategy_framework = create_winning_strategy_framework(
            full_framework, strategy_argument_names
        )

        self.create_graph_visualisation(
            framework=strategy_framework,
            output_path=output_path,
            title=title,
            layout=layout,
        )

    def create_graph_visualisation(
        self,
        framework: ArgumentationFramework,
        output_path: str,
        title: str | None = None,
        layout: str = "hierarchical",
        node_size: float = 0.3,
        font_size: int = 10,
    ) -> None:
        """Create a graph-based visualisation of the argumentation framework."""
        if title is None:
            title = f"Argumentation Framework: {framework.name}"

        arguments = framework.get_arguments()
        defeats = framework.get_defeats()

        if not arguments:
            print("No arguments to visualise")
            return

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_title(title, fontsize=16, fontweight="bold")

        if layout == "circular":
            positions = self._circular_layout(arguments)
        elif layout == "hierarchical":
            positions = self._hierarchical_layout(arguments, defeats)
        else:
            positions = self._spring_layout(arguments, defeats)

        for defeat in defeats:
            from_pos = positions[defeat.from_arg.name]
            to_pos = positions[defeat.to_arg.name]

            dx = to_pos[0] - from_pos[0]
            dy = to_pos[1] - from_pos[1]
            length = math.sqrt(dx * dx + dy * dy)

            if length > 0:

                unit_x = dx / length
                unit_y = dy / length

                start_x = from_pos[0] + node_size * unit_x
                start_y = from_pos[1] + node_size * unit_y
                end_x = to_pos[0] - node_size * unit_x
                end_y = to_pos[1] - node_size * unit_y

                ax.annotate(
                    "",
                    xy=(end_x, end_y),
                    xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.5, alpha=0.8),
                )

        for arg in arguments:
            pos = positions[arg.name]

            circle = patches.Circle(
                pos,
                node_size,
                facecolor="lightblue",
                edgecolor="darkblue",
                linewidth=2,
                alpha=0.9,
            )
            ax.add_patch(circle)

            ax.text(
                pos[0],
                pos[1],
                arg.name,
                ha="center",
                va="center",
                fontsize=font_size,
                fontweight="bold",
                wrap=True,
            )

        ax.set_aspect("equal")
        ax.axis("off")

        if positions:
            x_coords = [pos[0] for pos in positions.values()]
            y_coords = [pos[1] for pos in positions.values()]
            margin = max(node_size * 2, 0.5)
            ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
            ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _circular_layout(self, arguments: set) -> dict[str, tuple[float, float]]:
        """Create circular layout for arguments."""
        positions = {}
        arg_list = list(arguments)
        n = len(arg_list)

        if n == 0:
            return positions
        if n == 1:
            return {arg_list[0].name: (0, 0)}

        radius = 2.0
        angle_step = 2 * math.pi / n
        for i, arg in enumerate(arg_list):
            angle = i * angle_step
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            positions[arg.name] = (x, y)
        return positions

    def _hierarchical_layout(
        self, arguments: set, defeats: set, vertical_spacing: float = 2.5
    ) -> dict[str, tuple[float, float]]:
        """Create hierarchical (tree) layout with root at top moving downward."""
        if not arguments:
            return {}

        if len(arguments) == 1:
            arg = next(iter(arguments))
            return {arg.name: (0, 0)}

        arg_names = {arg.name for arg in arguments}
        attackers = {name: [] for name in arg_names}
        attacked_by = {name: [] for name in arg_names}

        for defeat in defeats:
            if defeat.from_arg.name in arg_names and defeat.to_arg.name in arg_names:
                attackers[defeat.to_arg.name].append(defeat.from_arg.name)
                attacked_by[defeat.from_arg.name].append(defeat.to_arg.name)

        roots = [name for name in arg_names if not attackers[name]]
        if not roots:

            roots = [next(iter(arg_names))]

        levels = {}
        queue = [(root, 0) for root in roots]
        visited = set()

        while queue:
            node, level = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            levels[node] = level

            for child in attacked_by[node]:
                if child not in visited:
                    queue.append((child, level + 1))

        max_level = max(levels.values()) if levels else 0
        for name in arg_names:
            if name not in levels:
                levels[name] = max_level + 1

        level_groups = {}
        for name, level in levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(name)

        positions = {}
        for level, nodes in level_groups.items():
            y = level * vertical_spacing
            n_nodes = len(nodes)

            if n_nodes == 1:
                positions[nodes[0]] = (0, y)
            else:

                total_width = (n_nodes - 1) * 2.0
                start_x = -total_width / 2
                for i, node in enumerate(nodes):
                    x = start_x + i * 2.0
                    positions[node] = (x, y)

        return positions

    def _spring_layout(
        self, arguments: set, defeats: set, iterations: int = 50
    ) -> dict[str, tuple[float, float]]:
        """Create spring-force layout."""
        if not arguments:
            return {}

        arg_list = list(arguments)
        n = len(arg_list)

        if n == 1:
            return {arg_list[0].name: (0, 0)}

        import random

        random.seed(42)
        positions = {}
        for arg in arg_list:
            positions[arg.name] = (random.uniform(-2, 2), random.uniform(-2, 2))

        connections = set()
        for defeat in defeats:
            connections.add((defeat.from_arg.name, defeat.to_arg.name))

        k = 1.0
        for _ in range(iterations):
            forces = {name: [0.0, 0.0] for name in positions}

            for i, name1 in enumerate(positions):
                for name2 in list(positions.keys())[i + 1 :]:
                    pos1 = positions[name1]
                    pos2 = positions[name2]
                    dx = pos1[0] - pos2[0]
                    dy = pos1[1] - pos2[1]
                    dist = math.sqrt(dx * dx + dy * dy)

                    if dist > 0:
                        force = k * k / dist
                        fx = force * dx / dist
                        fy = force * dy / dist
                        forces[name1][0] += fx
                        forces[name1][1] += fy
                        forces[name2][0] -= fx
                        forces[name2][1] -= fy

            for name1, name2 in connections:
                if name1 in positions and name2 in positions:
                    pos1 = positions[name1]
                    pos2 = positions[name2]
                    dx = pos2[0] - pos1[0]
                    dy = pos2[1] - pos1[1]
                    dist = math.sqrt(dx * dx + dy * dy)

                    if dist > 0:
                        force = dist * dist / k
                        fx = force * dx / dist
                        fy = force * dy / dist
                        forces[name1][0] += fx
                        forces[name1][1] += fy
                        forces[name2][0] -= fx
                        forces[name2][1] -= fy

            for name in positions:
                fx, fy = forces[name]
                positions[name] = (
                    positions[name][0] + 0.1 * fx,
                    positions[name][1] + 0.1 * fy,
                )

        return positions
