"""Export LaTeX tables from notebook results and compile the existing Beamer style."""
from pathlib import Path
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SLIDES = ROOT / "slides"
RESULTS = ROOT / "results"


def tex(text):
    replacements = {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#"}
    return "".join(replacements.get(c, c) for c in str(text))


def table(name):
    rows = json.loads((RESULTS / f"scenario_{name}.json").read_text())
    text = [r"\begin{tabular}{lrrr}\toprule Policy & Expected doses & km & Objective\\\midrule"]
    for row in rows:
        text.append(f"{tex(row['policy'])} & {row['expected_service']:.1f} & {row['km']:.1f} & {row['expected_objective']:.1f}" + r"\\")
    text.append(r"\bottomrule\end{tabular}")
    benchmark = next(row for row in rows if row["policy"] == "benchmark")
    text.append(r"\par\vspace{2mm}{\small Benchmark optimum interval: " +
                f"[{benchmark['expected_objective']:.2f}, {benchmark['expected_objective_upper_bound']:.2f}]" + "}")
    return "\n".join(text)


def main():
    env = json.loads((RESULTS / "environment.json").read_text())
    sample = json.loads((RESULTS / "training_example.json").read_text())
    example = [r"\begin{tabular}{rrrrr}\toprule $\log$ pop & Unvaccinated & Accessibility & Exposure & Served\\\midrule"]
    for row in sample:
        example.append(" & ".join(f"{row[key]:.2f}" for key in ["log population", "unvaccinated", "inaccessibility", "exposure", "served"]) + r"\\")
    example.append(r"\bottomrule\end{tabular}")
    environment = tex(env["cpu"]) + r"\\[2mm]" + (
        f"{env['physical_cores']} cores, {env['logical_cores']} logical processors; {env['ram_gib']:.1f} GiB RAM." + r"\\[2mm]" +
        tex(f"{env['os']}; Python {env['python']}; Gurobi {env['gurobi']}.") + r"\\[2mm]" +
        tex(f"scikit-learn {env['packages']['scikit-learn']}; gurobi-machinelearning {env['packages']['gurobi-machinelearning']}.") + r"\\[2mm]" +
        f"{env['threads']} solver threads; {env['settings']['time_limit']:g} s limit; relative MIP gap {env['settings']['mip_gap']:g}." + r"\\[2mm]" +
        tex("Run: " + env['utc'][:19] + " UTC; full versions, seeds and hashes in results/environment.json."))
    macros = "\n".join("\\newcommand{\\" + key + "}{\n" + value + "\n}" for key,value in
                       [("ScenarioA",table("a")),("ScenarioB",table("b")),("TrainingExample","\n".join(example)),("RunEnvironment",environment)])
    (SLIDES / "results.tex").write_text(macros + "\n", encoding="utf8")
    intro = SLIDES / "intro.tex"
    preamble = intro.read_text(encoding="utf8").split(r"\begin{document}")[0]
    intro.write_text(preamble + "\\begin{document}\n\\input{content.tex}\n\\end{document}\n", encoding="utf8")
    output = ROOT / "build" / "latex"
    output.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which("pdflatex")
    if not compiler:
        raise RuntimeError("pdflatex is required to rebuild slides/intro.pdf")
    for _ in range(2):
        subprocess.run([compiler,"-interaction=nonstopmode","-halt-on-error",f"-output-directory={output}","intro.tex"],cwd=SLIDES,check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    shutil.copy2(output / "intro.pdf", SLIDES / "intro.pdf")
    print("Updated slides/intro.pdf using the notebook's actual result tables.")


if __name__ == "__main__":
    main()
