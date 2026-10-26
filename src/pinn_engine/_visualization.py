"""Mermaid/Graphviz rendering of graph metadata; no execution or AD policy."""

from html import escape
from .quantities import Quantity


def source_node(graph, derivatives, quantity, external):
    if quantity in external:
        return external[quantity]
    node = graph.producer_node(quantity)
    return f"m{node[1]}" if node[0] == "model" else derivatives.node_id(node)


def quantity_label(graph, quantity: Quantity) -> str:
    """Return the semantic quantity name."""
    return quantity.name


def to_mermaid(graph, derivatives, direction: str = "LR") -> str:
    """Return a styled Mermaid diagram. Runtime/sample sizes are not inferred.

    Paste the returned text into a Mermaid code block to display it.
    Model numbers refer to their original position in graph.models.
    """
    if direction not in {"LR", "TB"}:
        raise ValueError("direction must be 'LR' or 'TB'")

    def label(text):
        # Mermaid's numeric entities prevent user labels being parsed as syntax.
        return "".join(f"#{ord(c)};" if c in '&<>"|#[]{}()' else c
                       for c in str(text).replace("\r", " ").replace("\n", " "))

    lines = [f"flowchart {direction}"]
    external = {q: f"i{i}" for i, q in enumerate(graph.inputs)}
    for q, node in external.items():
        lines.append(f'  {node}(["Input: {label(quantity_label(graph, q))}"]):::input')
    for index, model in enumerate(graph.models):
        name = label(type(model).__name__)
        lines.append(f'  m{index}["{index + 1}. {name}"]:::model')
    for index, derivative in enumerate(derivatives.requests):
        lines.append(f'  d{index}["{label(derivative.name)}"]:::derivative')
        for quantity, role in derivatives.graph_inputs(derivative):
            source = source_node(graph, derivatives, quantity, external)
            caption = label(f"{role}: {quantity_label(graph, quantity)}")
            lines.append(f'  {source} -->|"{caption}"| d{index}')
    for index, model in enumerate(graph.models):
        for q in model.inputs:
            source = source_node(graph, derivatives, q, external)
            lines.append(f'  {source} -->|"{label(quantity_label(graph, q))}"| m{index}')
    for index, q in enumerate(graph.outputs):
        lines.append(f'  o{index}(["Output: {label(quantity_label(graph, q))}"]):::output')
        lines.append(f'  {source_node(graph, derivatives, q, external)} --> o{index}')
    lines += [
        "  classDef input fill:#EFF6FF,stroke:#93C5FD,color:#1E40AF,stroke-width:1.5px;",
        "  classDef model fill:#F5F3FF,stroke:#A78BFA,color:#4C1D95,stroke-width:1.5px;",
        "  classDef output fill:#ECFDF5,stroke:#6EE7B7,color:#065F46,stroke-width:1.5px;",
        "  classDef derivative fill:#FFF7ED,stroke:#FB923C,color:#9A3412,stroke-width:1.5px;",
        "  linkStyle default stroke:#94A3B8,stroke-width:1.5px;",
    ]
    return "\n".join(lines)


def to_dot(graph, derivatives, direction: str = "LR", title: str = "Solver computation graph") -> str:
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
    external = {q: f"i{i}" for i, q in enumerate(graph.inputs)}
    for q, node in external.items():
        caption = card("EXTERNAL INPUT", quantity_label(graph, q), shape(q), "#2563EB")
        lines.append(f'  {node} [label={caption}, fillcolor="#EFF6FF", color="#93C5FD"];')
    # IDs stay tied to graph.models; layout is computed from dependencies.
    for index, model in enumerate(graph.models):
        sig = model.signature
        core = tuple(dict.fromkeys(a for side in (sig.inputs, sig.outputs)
                                   for axes in side.values() for a in axes))
        detail = "core axes: " + ", ".join(core) if core else "pointwise / no core sample axes"
        caption = card(f"MODEL {index + 1:02d}", type(model).__name__, detail, "#7C3AED")
        lines.append(f'  m{index} [label={caption}, fillcolor="#F5F3FF", color="#C4B5FD"];')
    for index, derivative in enumerate(derivatives.requests):
        detail = derivatives.graph_detail(derivative)
        caption = card("DERIVATIVE REQUEST", derivative.name, detail, "#C2410C")
        lines.append(f'  d{index} [label={caption}, fillcolor="#FFF7ED", color="#FB923C"];')
        for quantity, role in derivatives.graph_inputs(derivative):
            source = source_node(graph, derivatives, quantity, external)
            caption = quote(f"{role}: {quantity_label(graph, quantity)}")
            lines.append(f"  {source} -> d{index} [label={caption}, style=dashed];")
    for index, model in enumerate(graph.models):
        for q in model.inputs:
            source = source_node(graph, derivatives, q, external)
            # Input nodes already name the quantity; only internal arrows need it.
            attr = "" if q in external else f" [label={quote(quantity_label(graph, q))}]"
            lines.append(f"  {source} -> m{index}{attr};")
    for index, q in enumerate(graph.outputs):
        caption = card("RETURNED OUTPUT", quantity_label(graph, q), shape(q), "#059669")
        lines.append(f'  o{index} [label={caption}, fillcolor="#ECFDF5", color="#6EE7B7"];')
        lines.append(f'  {source_node(graph, derivatives, q, external)} -> o{index} [color="#34A88A"];')
    if external:
        lines.append("  { rank=same; " + "; ".join(external.values()) + "; }")
    lines.append("  { rank=same; " + "; ".join(f"o{i}" for i in range(len(graph.outputs))) + "; }")
    lines.append("}")
    return "\n".join(lines)


def render_graph(
    graph, derivatives, filename="solver_graph.svg", *, direction: str = "LR",
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
    source = to_dot(graph, derivatives, direction=direction, title=title)
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
