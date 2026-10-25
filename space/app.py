"""A demo of laya-kindynos-causal: who or what causes a written AI risk, whether the harm is intended, and
when it occurs, along the three axes of the MIT Causal Taxonomy of AI Risks.

The Space of MicheleStingo/laya-kindynos-causal on Hugging Face, published from space/ in the Kindynos
repository. Laya reads the risk as Kindynos writes it (agents/causality_analyzer/laya_classifier.py):
the subdomain, the title, the description; space/parity.py checks that both give the same labels.
"""

import base64
import html
import json
from pathlib import Path

import gradio as gr
import laya
from huggingface_hub import hf_hub_download

HERE = Path(__file__).parent
REPO, REVISION = "MicheleStingo/laya-kindynos-causal", "v1.0"
AXES = ("entity", "intent", "timing")
MAX_TEXT = 4000  # characters: a risk is a few sentences

agent = laya.load(REPO, revision=REVISION)
with open(hf_hub_download(REPO, "questions.json", revision=REVISION), encoding="utf-8") as f:
    QUESTIONS = json.load(f)  # the three axes as typed questions, with the MIT definitions
DATA = json.loads((HERE / "data.json").read_text(encoding="utf-8"))
SUBDOMAINS = DATA["subdomains"]  # code -> name, the MIT Domain Taxonomy

AXIS_NAMES = {"entity": "Entity", "intent": "Intent", "timing": "Timing"}
VALUE_NAMES = {
    "ai": "AI",
    "human": "Human",
    "other": "Other",
    "intentional": "Intentional",
    "unintentional": "Unintentional",
    "pre-deployment": "Pre-deployment",
    "post-deployment": "Post-deployment",
}


def state(subdomain: str | None, title: str, text: str) -> str:
    """What Laya reads, as Kindynos writes it: the subdomain (code and name) when known, the title, the
    description."""
    lines = [f"Risk subdomain: {subdomain} {SUBDOMAINS[subdomain]}"] if subdomain in SUBDOMAINS else []
    return "\n".join(lines + ([title] if title else []) + [text])


def predict(subdomain: str | None, title: str, text: str) -> dict:
    """{axis: {"value", "probabilities"}} for one risk."""
    answers = agent.predict(state(subdomain, title, text), QUESTIONS)["answers"]
    return {ax: {"value": answers[ax]["choice"], "probabilities": answers[ax]["probabilities"]} for ax in AXES}


def _cause(axis: str, answer: dict) -> str:
    value, probabilities = answer["value"], answer["probabilities"]
    definition = QUESTIONS[axis]["criteria"][value]
    dot = f'<span class="dot {value}" aria-hidden="true"></span>' if axis == "entity" and value in ("ai", "human") else ""
    rows = "".join(
        f'<li class="{"chosen" if name == value else ""}"><span class="name">{VALUE_NAMES[name]}</span>'
        f'<span class="track"><span class="fill" style="width:{p * 100:.1f}%"></span></span>'
        f'<span class="share">{round(p * 100)}%</span></li>'
        for name, p in ((name, probabilities[name]) for name in QUESTIONS[axis]["criteria"])  # the taxonomy order
    )
    return (
        f'<article class="cause"><p class="axis">{AXIS_NAMES[axis]}</p>'
        f'<p class="question">{html.escape(QUESTIONS[axis]["instructions"])}</p>'
        f'<p class="value">{dot}{VALUE_NAMES[value]}</p>'
        f'<p class="definition">{html.escape(definition)}.</p>'
        f'<ul class="shares" aria-label="Probability of each value">{rows}</ul></article>'
    )


def render(result: dict) -> str:
    return '<div class="causes">' + "".join(_cause(ax, result[ax]) for ax in AXES) + "</div>"


EMPTY = '<div class="causes empty"><p>Write a risk, or pick an example, and its causes appear here.</p></div>'


def label(subdomain: str | None, title: str, text: str) -> tuple[str, dict]:
    title, text = (title or "").strip(), (text or "").strip()
    if not text:
        raise gr.Error("Write the description of the risk.")
    result = predict(subdomain or None, title, text[:MAX_TEXT])
    return render(result), result


def _data_uri(path: Path, kind: str) -> str:
    return f"data:{kind};base64," + base64.b64encode(path.read_bytes()).decode()


HEADER = f"""
<header class="masthead">
  <a class="brand" href="https://github.com/6MicheleStingo9/kindynos" target="_blank" rel="noopener">
    <img class="wordmark light" src="{_data_uri(HERE / "assets" / "kindynos-wordmark-light.svg", "image/svg+xml")}" alt="Kindynos">
    <img class="wordmark dark" src="{_data_uri(HERE / "assets" / "kindynos-wordmark-dark.svg", "image/svg+xml")}" alt="Kindynos">
  </a>
  <nav>
    <a href="https://huggingface.co/{REPO}" target="_blank" rel="noopener">Model</a>
    <a href="https://huggingface.co/datasets/MicheleStingo/kindynos-causal-risks" target="_blank" rel="noopener">Data</a>
    <a href="https://github.com/6MicheleStingo9/kindynos" target="_blank" rel="noopener">Kindynos</a>
  </nav>
</header>
<section class="hero">
  <div>
    <p class="eyebrow">laya-kindynos-causal</p>
    <h1>Who or what causes an AI risk?</h1>
    <p class="lead">Describe an AI risk and Laya labels its causes along the three axes of the
    <a href="https://airisk.mit.edu/" target="_blank" rel="noopener">MIT Causal Taxonomy</a>: who or what
    causes it, whether the harm is intended, and when it occurs, with a calibrated probability for each value.
    It reads English and Italian, and runs on a CPU.</p>
  </div>
  <img class="owl" src="{_data_uri(HERE / "assets" / "owl.png", "image/png")}" alt="">
</section>
"""

FOOTER = """
<footer class="colophon">
  <p>The labels come from a model: they support, but do not replace, the judgement of qualified people.
  The text you write is not stored.</p>
  <p>Laya-multilingual by ConvAI Innovations, fine-tuned for <a href="https://github.com/6MicheleStingo9/kindynos"
  target="_blank" rel="noopener">Kindynos</a> (Apache-2.0) · Taxonomies: MIT AI Risk Repository (Slattery et al.,
  2024), CC BY 4.0</p>
</footer>
"""

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght,SOFT@9..144,400..700,0..100'
    '&family=Inter:wght@400;500;600&display=swap">'
)

# The colours of the Kindynos interface (frontend/ui/src/styles/tokens.css), light and dark
THEME = gr.themes.Base(
    font=["Inter", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
    font_mono=["ui-monospace", "SF Mono", "Menlo", "Consolas", "monospace"],
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill="#f6f4ee",
    body_background_fill_dark="#0d1626",
    body_text_color="#132747",
    body_text_color_dark="#f4ebd9",
    body_text_color_subdued="#5d687e",
    body_text_color_subdued_dark="#9da8be",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#14213a",
    background_fill_secondary="#efece3",
    background_fill_secondary_dark="#1b2b48",
    block_background_fill="#ffffff",
    block_background_fill_dark="#14213a",
    block_border_color="#dfd9cb",
    block_border_color_dark="#2b3d5e",
    border_color_primary="#dfd9cb",
    border_color_primary_dark="#2b3d5e",
    block_label_text_color="#46536b",
    block_label_text_color_dark="#c6cddb",
    block_title_text_color="#132747",
    block_title_text_color_dark="#f4ebd9",
    input_background_fill="#ffffff",
    input_background_fill_dark="#0d1626",
    input_border_width="1px",
    input_border_width_dark="1px",
    input_border_color="#7f889c",
    input_border_color_dark="#6b7c9e",
    input_border_color_focus="#4b70b8",
    input_border_color_focus_dark="#8eaee3",
    button_primary_background_fill="#132747",
    button_primary_background_fill_dark="#fae5b9",
    button_primary_background_fill_hover="#24395c",
    button_primary_background_fill_hover_dark="#f3d79c",
    button_primary_text_color="#fae5b9",
    button_primary_text_color_dark="#132747",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_dark="#14213a",
    button_secondary_text_color="#132747",
    button_secondary_text_color_dark="#f4ebd9",
    table_even_background_fill="#ffffff",
    table_even_background_fill_dark="#14213a",
    table_odd_background_fill="#faf8f3",
    table_odd_background_fill_dark="#182740",
    table_border_color="#dfd9cb",
    table_border_color_dark="#2b3d5e",
    table_text_color="#132747",
    table_text_color_dark="#f4ebd9",
    color_accent="#4b70b8",
    color_accent_soft="#e8eef8",
    color_accent_soft_dark="#22324f",
    link_text_color="#132747",
    link_text_color_dark="#fae5b9",
    link_text_color_visited="#132747",
    link_text_color_visited_dark="#fae5b9",
)

CSS = (HERE / "style.css").read_text(encoding="utf-8")

with gr.Blocks(title="Laya · who or what causes an AI risk") as demo:
    gr.HTML(HEADER)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=320):
            subdomain = gr.Dropdown(
                choices=[("Not specified", "")] + [(f"{code} {name}", code) for code, name in SUBDOMAINS.items()],
                value="",
                label="Subdomain of the MIT Domain Taxonomy",
                info="Optional: Laya reads it with the text.",
            )
            title = gr.Textbox(label="Title", placeholder="Generation of insecure or malicious code", max_lines=2)
            text = gr.Textbox(
                label="Description",
                placeholder="Describe the risk in a few sentences, in English or Italian.",
                lines=6,
                max_length=MAX_TEXT,
            )
            run = gr.Button("Label the causes", variant="primary")
            # The examples under the button: in view on a desktop, before the results on a phone
            causes = gr.HTML(EMPTY, render=False)
            answer = gr.JSON(label="Labels and probabilities", show_label=False, render=False)
            gr.Examples(
                examples=[[e["subdomain"], e["title"], e["text"]] for e in DATA["examples"]],
                inputs=[subdomain, title, text],
                outputs=[causes, answer],
                fn=label,
                run_on_click=True,
                cache_examples=False,
                label="Or pick a risk written by Kindynos, from its test set",
                example_labels=[e["title"] for e in DATA["examples"]],
                examples_per_page=6,
            )
        with gr.Column(scale=6, min_width=320):
            causes.render()
            with gr.Accordion("As JSON", open=False):
                answer.render()
    gr.HTML(FOOTER)
    run.click(label, inputs=[subdomain, title, text], outputs=[causes, answer], api_name="label")

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, head=FONTS)
