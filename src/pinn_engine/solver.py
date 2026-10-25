"""Dependency graph execution and visualization for Models and derivatives."""

from __future__ import annotations
from html import escape
from typing import Mapping, Sequence
from torch import nn
from .quantities import Quantity
from .model import Model
from .values import Value, _check_value
from .derivatives import (
    Derivative, DerivativePlan, _related_quantities,
    plan_derivative, evaluate_derivative, validate_autograd,
)

class Solver(nn.Module):
    """Wire Models and symbolic first derivatives in one hosting graph.

    Derivatives are virtual producer nodes, never external Tensor inputs and
    never user Model outputs. Plans identify source models and held cut points.
    Numerical evaluation of every derivative deliberately raises NotImplemented.

    Selected outputs determine the executed ancestors. A forward outputs=
    override permits inference and unfinished physics in the same Solver.
    There is no numerical derivative backend, grouping, persistent cache, or
    scheduling optimization here. The default derivative boundary checks AD
    compatibility. Ordinary outputs retain the v2 named-axis execution.
    """

    def __init__(
        self, models: Sequence[Model],
        outputs: Quantity | Sequence[Quantity] | None = None,
    ) -> None:
        super().__init__()
        self.models = tuple(models)
        if not self.models or any(not isinstance(model, Model) for model in self.models):
            raise ValueError("Provide at least one Model")

        self._quantities = {}
        produced, required = {}, {}
        derivatives = {}
        self.geometry = None
        for index, model in enumerate(self.models):
            signature = model.signature
            if not signature.outputs:
                raise ValueError("Each connected model must declare an output")
            for side in (signature.inputs, signature.outputs):
                for quantity in side:
                    self._register_quantity(quantity)
                    if isinstance(quantity, Derivative):
                        derivatives[quantity] = None
            for quantity in signature.outputs:
                if isinstance(quantity, Derivative):
                    raise ValueError("Derivative quantities are produced by Solver, not by Models")
                if quantity in produced:
                    raise ValueError(f"Multiple producers for {quantity.name!r}")
                produced[quantity] = index
            required.update(dict.fromkeys(signature.inputs))

            # Keep the existing same-domain rule; geometric subset policy is open.
            if model.geometry is not None:
                if self.geometry is not None and model.geometry is not self.geometry:
                    raise ValueError("Connected models must share one geometry object")
                self.geometry = model.geometry
            self.add_module(f"part_{index}", model)

        # A derivative can be requested directly, without a residual consumer.
        if outputs is not None:
            outputs = (outputs,) if isinstance(outputs, Quantity) else tuple(outputs)
            for quantity in outputs:
                if not isinstance(quantity, Quantity):
                    raise TypeError("Solver outputs must be Quantity objects")
                self._register_quantity(quantity)
                if isinstance(quantity, Derivative):
                    derivatives[quantity] = None
        self.derivatives = tuple(derivatives)
        for derivative in self.derivatives:
            required.update(dict.fromkeys((derivative.of, derivative.wrt,
                                           *(derivative.holding or ()))))
        provided = set(produced) | set(self.derivatives)
        self.inputs = tuple(q for q in required if q not in provided)
        selected = tuple(produced) if outputs is None else (
            (outputs,) if isinstance(outputs, Quantity) else tuple(outputs)
        )
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Select at least one distinct forward output")
        for quantity in selected:
            if quantity not in provided:
                raise ValueError(f"Unknown forward output: {quantity!r}")
            if quantity.shape != self._quantities[quantity].shape:
                raise ValueError(f"Conflicting shapes for {quantity.name!r}")
        self.outputs = selected
        self._producers = produced

        # Both ordinary models and derivative requests participate in ordering.
        # The latter must precede residual consumers, but follow source values.
        self._node_dependencies = {
            ("model", index): tuple(model.inputs)
            for index, model in enumerate(self.models)
        }
        for derivative in self.derivatives:
            self._node_dependencies[("derivative", derivative)] = tuple(dict.fromkeys(
                (derivative.of, derivative.wrt, *(derivative.holding or ()))
            ))
        available, pending = set(self.inputs), dict(self._node_dependencies)
        self._ordered = []
        while pending:
            ready = [node for node, dependencies in pending.items()
                     if set(dependencies) <= available]
            if not ready:
                raise ValueError("Cyclic model dependencies; use distinct input/output quantities")
            for node in ready:
                self._ordered.append(node)
                kind, key = node
                available.update(self.models[key].outputs if kind == "model" else (key,))
                del pending[node]

        # Planning is independent of actual Tensor sizes and the AD/FD backend.
        self.derivative_plans = {
            derivative: self._plan_derivative(derivative)
            for derivative in self.derivatives
        }

    def _register_quantity(self, quantity: Quantity) -> None:
        if not isinstance(quantity, Quantity):
            raise TypeError("Expected Quantity objects")
        for related in _related_quantities(quantity):
            previous = self._quantities.get(related)
            if previous is not None and previous.shape != related.shape:
                raise ValueError(f"Conflicting shapes for {related.name!r}")
            self._quantities[related] = related

    def _plan_derivative(self, derivative: Derivative) -> DerivativePlan:
        """Delegate derivative path planning to the derivative module."""
        return plan_derivative(self, derivative)

    def _producer_node(self, quantity: Quantity):
        if isinstance(quantity, Derivative):
            return ("derivative", quantity)
        if quantity in self._producers:
            return ("model", self._producers[quantity])
        return None

    def _needed_nodes(self, outputs):
        needed = set()

        def visit(quantity):
            node = self._producer_node(quantity)
            if node is None or node in needed:
                return
            needed.add(node)
            for dependency in self._node_dependencies[node]:
                visit(dependency)

        for quantity in outputs:
            visit(quantity)
        return needed

    def _evaluate_derivative(
        self, plan: DerivativePlan, values: Mapping[Quantity, Value],
    ) -> Value:
        """Derivative execution hook; delegate to the default backend boundary."""
        return evaluate_derivative(self, plan, values)

    def validate_autograd(
        self, plan: DerivativePlan, values: Mapping[Quantity, Value],
    ) -> None:
        """Delegate default AD capability checks to the derivative module."""
        validate_autograd(self, plan, values)

    def forward(
        self, values: Mapping[Quantity, Value], *,
        outputs: Quantity | Sequence[Quantity] | None = None,
    ) -> dict[Quantity, Value]:
        """Execute selected ancestors; unfinished derivatives fail explicitly."""
        if set(values) != set(self.inputs):
            raise ValueError(f"Expected input quantities: {[q.name for q in self.inputs]}")
        selected = self.outputs if outputs is None else (
            (outputs,) if isinstance(outputs, Quantity) else tuple(outputs)
        )
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Select at least one distinct forward output")
        for quantity in selected:
            if quantity not in self._producers and quantity not in self.derivative_plans:
                raise ValueError(f"Unknown forward output: {quantity!r}")
            if quantity.shape != self._quantities[quantity].shape:
                raise ValueError(f"Conflicting shapes for {quantity.name!r}")
        pool = dict(values)
        sizes = {}
        for quantity, value in values.items():
            if quantity.shape != self._quantities[quantity].shape:
                raise ValueError(f"Conflicting shapes for {quantity.name!r}")
            _check_value(quantity, value, sizes)

        needed = self._needed_nodes(selected)
        for node in self._ordered:
            if node not in needed:
                continue
            kind, key = node
            if kind == "derivative":
                value = self._evaluate_derivative(self.derivative_plans[key], pool)
                _check_value(key, value, sizes)
                pool[key] = value
                continue
            model = self.models[key]
            signature = model.signature
            inputs = {quantity: pool[quantity] for quantity in signature.inputs}
            # Calling the module preserves hooks and each model's own vmap.
            result = model(inputs)
            if set(result) != set(signature.outputs):
                raise ValueError("Model returned quantities different from its signature")
            for quantity, value in result.items():
                _check_value(quantity, value, sizes)
                pool[quantity] = value
        return {quantity: pool[quantity] for quantity in selected}

    def _source_node(self, quantity: Quantity, external) -> str:
        if quantity in external:
            return external[quantity]
        if isinstance(quantity, Derivative):
            return f"d{self.derivatives.index(quantity)}"
        return f"m{self._producers[quantity]}"

    def _graph_quantity_label(self, quantity: Quantity) -> str:
        """Disambiguate equal names on different domains without object ids."""
        label = quantity.name
        matches = [q for q in self._quantities if q.name == quantity.name]
        if len(matches) > 1:
            domains = []
            for q in matches:
                if all(q._domain is not d for d in domains):
                    domains.append(q._domain)
            if quantity._domain is None:
                label += " [nonspatial]"
            else:
                number = next(i for i, d in enumerate(domains, 1)
                              if d is quantity._domain)
                label += f" [domain {number}]"
        return label

    def to_mermaid(self, direction: str = "LR") -> str:
        """Return a styled Mermaid diagram. Runtime/sample sizes are not inferred.

        Paste the returned text into a Mermaid code block to display it.
        Model numbers refer to their original position in self.models.
        """
        if direction not in {"LR", "TB"}:
            raise ValueError("direction must be 'LR' or 'TB'")

        def label(text):
            # Mermaid's numeric entities prevent user labels being parsed as syntax.
            return "".join(f"#{ord(c)};" if c in '&<>"|#[]{}()' else c
                           for c in str(text).replace("\r", " ").replace("\n", " "))

        lines = [f"flowchart {direction}"]
        external = {q: f"i{i}" for i, q in enumerate(self.inputs)}
        for q, node in external.items():
            lines.append(f'  {node}(["Input: {label(self._graph_quantity_label(q))}"]):::input')
        for index, model in enumerate(self.models):
            name = label(type(model).__name__)
            lines.append(f'  m{index}["{index + 1}. {name}"]:::model')
        for index, derivative in enumerate(self.derivatives):
            lines.append(f'  d{index}["{label(derivative.name)}"]:::derivative')
            for quantity, role in self._derivative_graph_inputs(derivative):
                source = self._source_node(quantity, external)
                caption = label(f"{role}: {self._graph_quantity_label(quantity)}")
                lines.append(f'  {source} -->|"{caption}"| d{index}')
        for index, model in enumerate(self.models):
            for q in model.inputs:
                source = self._source_node(q, external)
                lines.append(f'  {source} -->|"{label(self._graph_quantity_label(q))}"| m{index}')
        for index, q in enumerate(self.outputs):
            lines.append(f'  o{index}(["Output: {label(self._graph_quantity_label(q))}"]):::output')
            lines.append(f'  {self._source_node(q, external)} --> o{index}')
        lines += [
            "  classDef input fill:#EFF6FF,stroke:#93C5FD,color:#1E40AF,stroke-width:1.5px;",
            "  classDef model fill:#F5F3FF,stroke:#A78BFA,color:#4C1D95,stroke-width:1.5px;",
            "  classDef output fill:#ECFDF5,stroke:#6EE7B7,color:#065F46,stroke-width:1.5px;",
            "  classDef derivative fill:#FFF7ED,stroke:#FB923C,color:#9A3412,stroke-width:1.5px;",
            "  linkStyle default stroke:#94A3B8,stroke-width:1.5px;",
        ]
        return "\n".join(lines)

    @staticmethod
    def _derivative_graph_inputs(derivative: Derivative):
        yield derivative.of, "of"
        yield derivative.wrt, "wrt"
        for quantity in derivative.holding or ():
            yield quantity, "holding"

    def to_dot(self, direction: str = "LR", title: str = "Solver computation graph") -> str:
        """Return Graphviz DOT with automatic layout and vector-quality labels.

        Blue: external inputs. Purple: models. Green: selected returned outputs.
        Core axes are signature axes, not runtime batch axes. No model executes.
        """
        import json
        from textwrap import wrap

        if direction not in {"LR", "TB"}:
            raise ValueError("direction must be 'LR' or 'TB'")

        def quote(text):
            return json.dumps(str(text), ensure_ascii=False)

        def html(text):
            return escape(str(text), quote=True)

        def wrapped(text, width=30):
            return '<BR/>'.join(html(line) for line in
                               wrap(str(text), width, break_long_words=True) or [""])

        def card(kicker, heading, detail, color):
            return ('<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3">'
                    f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9" COLOR="{color}">{html(kicker)}</FONT></TD></TR>'
                    f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="15"><B>{wrapped(heading)}</B></FONT></TD></TR>'
                    f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#64748B">{wrapped(detail, 42)}</FONT></TD></TR>'
                    '</TABLE>>')

        def shape(q):
            return "scalar" if not q.shape else "shape: " + " x ".join(map(str, q.shape))

        lines = [
            "digraph Solver {",
            f"  graph [rankdir={direction}, bgcolor=\"#FFFFFF\", pad=0.45, nodesep=0.45,",
            "         ranksep=0.85, splines=spline, outputorder=edgesfirst,",
            "         fontname=\"DejaVu Sans\", fontsize=22, fontcolor=\"#0F172A\",",
            f"         labelloc=t, labeljust=l, label={quote(title)}];",
            "  node [shape=box, style=\"rounded,filled\", fontname=\"DejaVu Sans\",",
            "        fontcolor=\"#0F172A\", penwidth=1.3, margin=\"0.20,0.14\"];",
            "  edge [color=\"#94A3B8\", penwidth=1.5, arrowsize=0.7,",
            "        fontname=\"DejaVu Sans\", fontsize=10, fontcolor=\"#475569\"];",
        ]
        external = {q: f"i{i}" for i, q in enumerate(self.inputs)}
        for q, node in external.items():
            caption = card("EXTERNAL INPUT", self._graph_quantity_label(q), shape(q), "#2563EB")
            lines.append(f'  {node} [label={caption}, fillcolor="#EFF6FF", color="#93C5FD"];')
        # IDs stay tied to self.models; layout is computed from dependencies.
        for index, model in enumerate(self.models):
            sig = model.signature
            core = tuple(dict.fromkeys(a for side in (sig.inputs, sig.outputs)
                                       for axes in side.values() for a in axes))
            detail = "core axes: " + ", ".join(core) if core else "pointwise / no core sample axes"
            caption = card(f"MODEL {index + 1:02d}", type(model).__name__, detail, "#7C3AED")
            lines.append(f'  m{index} [label={caption}, fillcolor="#F5F3FF", color="#C4B5FD"];')
        for index, derivative in enumerate(self.derivatives):
            plan = self.derivative_plans[derivative]
            detail = f"planned source models: {plan.model_indices}; execution not implemented"
            caption = card("DERIVATIVE REQUEST", derivative.name, detail, "#C2410C")
            lines.append(f'  d{index} [label={caption}, fillcolor="#FFF7ED", color="#FB923C"];')
            for quantity, role in self._derivative_graph_inputs(derivative):
                source = self._source_node(quantity, external)
                caption = quote(f"{role}: {self._graph_quantity_label(quantity)}")
                lines.append(f"  {source} -> d{index} [label={caption}, style=dashed];")
        for index, model in enumerate(self.models):
            for q in model.inputs:
                source = self._source_node(q, external)
                # Input nodes already name the quantity; only internal arrows need it.
                attr = "" if q in external else f" [label={quote(self._graph_quantity_label(q))}]"
                lines.append(f"  {source} -> m{index}{attr};")
        for index, q in enumerate(self.outputs):
            caption = card("RETURNED OUTPUT", self._graph_quantity_label(q), shape(q), "#059669")
            lines.append(f'  o{index} [label={caption}, fillcolor="#ECFDF5", color="#6EE7B7"];')
            lines.append(f'  {self._source_node(q, external)} -> o{index} [color="#34A88A"];')
        if external:
            lines.append("  { rank=same; " + "; ".join(external.values()) + "; }")
        lines.append("  { rank=same; " + "; ".join(f"o{i}" for i in range(len(self.outputs))) + "; }")
        lines.append("}")
        return "\n".join(lines)

    def render_graph(
        self, filename="solver_graph.svg", *, direction: str = "LR",
        title: str = "Solver computation graph",
    ):
        """Save .svg/.png/.pdf using the Graphviz 'dot' executable; return Path.

        Install the Graphviz system application first (not just its Python package).
        SVG is recommended for zooming; PNG is rendered at 180 dpi.
        Example: solver.render_graph("solver_graph.svg")
        """
        from pathlib import Path
        import shutil
        import subprocess

        path = Path(filename).expanduser()
        fmt = path.suffix.lower().lstrip(".")
        if fmt not in {"svg", "png", "pdf"}:
            raise ValueError("filename must end in .svg, .png, or .pdf")
        source = self.to_dot(direction=direction, title=title)
        executable = shutil.which("dot")
        if executable is None:
            raise RuntimeError(
                "Graphviz 'dot' was not found. Install Graphviz and add its bin folder "
                "to PATH. On Colab/Ubuntu: !apt-get -qq install graphviz. "
                "You can still use solver.to_mermaid() without Graphviz."
            )
        result = subprocess.run(
            [executable, f"-T{fmt}", "-Gdpi=180"], input=source.encode("utf-8"),
            capture_output=True, check=False,
        )
        if result.returncode:
            raise RuntimeError("Graphviz could not render the graph: "
                               + result.stderr.decode("utf-8", errors="replace"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(result.stdout)
        return path
