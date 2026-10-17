"""Run one short deterministic smoke test of the copied simulator."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from aircraft_sim import DetectAction, DetectEnv


def main() -> None:
    env = DetectEnv()
    observations = env.reset({"seed": 1, "k_red": 2})
    for _ in range(5):
        result = env.step({i: DetectAction() for i in range(env.n)})
        observations = result.observations
        if result.done:
            break
    print({
        "agents": env.n,
        "observation_agents": sorted(observations),
        "steps": env.step_count,
        "done": result.done,
        "terminal": result.info.get("terminal"),
        "reward": result.reward,
    })


if __name__ == "__main__":
    main()
