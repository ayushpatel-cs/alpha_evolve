import os
from openevolve import OpenEvolve

# Ensure API key is set
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY environment variable")

# Initialize the system
evolve = OpenEvolve(
    initial_program_path="initial_program.py",
    evaluation_file="evaluator.py",
    config_path="config_phase_1.yaml"
)

# Run the evolution
best_program = evolve.run(iterations=1000)
print(f"Best program metrics:")
for name, value in best_program.metrics.items():
    print(f"  {name}: {value:.4f}")