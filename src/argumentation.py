from dataclasses import dataclass

from py_arg.abstract_argumentation_classes.abstract_argumentation_framework import (
    AbstractArgumentationFramework,
)
from py_arg.abstract_argumentation_classes.argument import Argument
from py_arg.abstract_argumentation_classes.defeat import Defeat
from py_arg.algorithms.semantics.get_grounded_extension import get_grounded_extension


@dataclass(frozen=True)
class ArgumentWrapper:
    """Wrapper for PyArg Argument with additional metadata for AF-CBA."""

    name: str
    content: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "argument", Argument(self.name))

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, ArgumentWrapper):
            return self.name == other.name
        return False

    def __repr__(self):
        return f"ArgumentWrapper('{self.name}')"


@dataclass(frozen=True)
class DefeatWrapper:
    """Wrapper for PyArg Defeat with additional metadata."""

    from_arg: ArgumentWrapper
    to_arg: ArgumentWrapper

    def __post_init__(self):
        object.__setattr__(
            self, "defeat", Defeat(self.from_arg.argument, self.to_arg.argument)
        )

    def __repr__(self):
        return f"DefeatWrapper({self.from_arg.name} -> {self.to_arg.name})"


class ArgumentationFramework:
    """High-level interface for constructing and analysing argumentation frameworks."""

    def __init__(self, name: str = "af"):
        self.name = name
        self._arguments: set[ArgumentWrapper] = set()
        self._defeats: set[DefeatWrapper] = set()
        self._pyarg_framework: AbstractArgumentationFramework | None = None
        self._grounded_extension: set[ArgumentWrapper] | None = None

    def clear(self):
        """Clear all arguments, defeats, and caches."""
        self._arguments.clear()
        self._defeats.clear()
        self._invalidate_cache()

    def add_argument(
        self, arg_name: str, content: str | None = None
    ) -> ArgumentWrapper:
        """Add an argument to the framework."""
        if any(arg.name == arg_name for arg in self._arguments):
            raise ValueError(f"Argument '{arg_name}' already exists")

        arg = ArgumentWrapper(arg_name, content)
        self._arguments.add(arg)
        self._invalidate_cache()
        return arg

    def add_defeat(self, from_arg: str, to_arg: str) -> DefeatWrapper:
        """Add a defeat relation between two arguments."""
        from_wrapper = self.get_argument(from_arg)
        to_wrapper = self.get_argument(to_arg)

        if from_wrapper is None:
            raise ValueError(f"Argument '{from_arg}' not found")
        if to_wrapper is None:
            raise ValueError(f"Argument '{to_arg}' not found")

        defeat = DefeatWrapper(from_wrapper, to_wrapper)
        self._defeats.add(defeat)
        self._invalidate_cache()
        return defeat

    def get_argument(self, arg_name: str) -> ArgumentWrapper | None:
        """Retrieve an argument by name."""
        for arg in self._arguments:
            if arg.name == arg_name:
                return arg
        return None

    def get_arguments(self) -> set[ArgumentWrapper]:
        """Get all arguments in the framework."""
        return self._arguments.copy()

    def get_defeats(self) -> set[DefeatWrapper]:
        """Get all defeat relations in the framework."""
        return self._defeats.copy()

    def get_grounded_extension(self) -> set[ArgumentWrapper]:
        """Compute and return the grounded extension."""
        if self._grounded_extension is None:
            self._compute_grounded_extension()
        return self._grounded_extension.copy()

    def _invalidate_cache(self):
        """Invalidate cached computations when framework changes."""
        self._pyarg_framework = None
        self._grounded_extension = None

    def _build_pyarg_framework(self):
        """Build the underlying PyArg framework."""
        if self._pyarg_framework is None:
            pyarg_args = [arg.argument for arg in self._arguments]
            pyarg_defeats = [defeat.defeat for defeat in self._defeats]
            self._pyarg_framework = AbstractArgumentationFramework(
                self.name, pyarg_args, pyarg_defeats
            )

    def _compute_grounded_extension(self):
        """Compute the grounded extension using PyArg."""
        self._build_pyarg_framework()
        pyarg_grounded = get_grounded_extension(self._pyarg_framework)

        self._grounded_extension = set()
        for pyarg_arg in pyarg_grounded:
            for wrapper in self._arguments:
                if wrapper.argument == pyarg_arg:
                    self._grounded_extension.add(wrapper)
                    break

    def is_in_grounded(self, arg_name: str) -> bool:
        """Check if an argument is in the grounded extension."""
        arg = self.get_argument(arg_name)
        if arg is None:
            return False
        return arg in self.get_grounded_extension()

    def get_attackers(self, arg_name: str) -> set[ArgumentWrapper]:
        """Get all arguments that attack the given argument."""
        attackers = set()
        for defeat in self._defeats:
            if defeat.to_arg.name == arg_name:
                attackers.add(defeat.from_arg)
        return attackers

    def get_attacked_by(self, arg_name: str) -> set[ArgumentWrapper]:
        """Get all arguments attacked by the given argument."""
        attacked = set()
        for defeat in self._defeats:
            if defeat.from_arg.name == arg_name:
                attacked.add(defeat.to_arg)
        return attacked

    def __repr__(self):
        return (
            f"ArgumentationFramework('{self.name}', "
            f"{len(self._arguments)} arguments, "
            f"{len(self._defeats)} defeats)"
        )
