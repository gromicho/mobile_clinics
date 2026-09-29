"""Check policy results against valid simulator bounds, not assumed rankings."""
from pathlib import Path
import pandas as pd

for filename, groups in [("scenario_a", ["gamma"]), ("scenario_b", ["gamma"]),
                         ("sweep", ["gamma"]), ("parameter_experiments", ["experiment"]),
                         ("history_sensitivity", ["gamma", "history_seed"])]:
    frame = pd.read_json(Path("results") / f"{filename}.json")
    for _, group in frame.groupby(groups):
        benchmark = group.loc[group.policy.eq("benchmark")].iloc[0]
        assert group.expected_objective.max() <= benchmark.expected_objective_upper_bound + 1e-5
        assert (group.status == 2).all(), "Teaching snapshot must solve all models to their configured tolerance"
        assert group.mip_gap.max() <= 1e-6 + 1e-8
        assert benchmark.benchmark_interpolation_error <= .5 + 1e-8
        assert (group.variables <= 2000).all() and (group.linear_constraints <= 2000).all()
print("All policies respect the simulator upper bounds and teaching model-size limits.")
