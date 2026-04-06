import traci

traci.start(["sumo-gui", "-c", "config.sumocfg"])

for step in range(1000):
    traci.simulationStep()

    # Get current phase
    phase = traci.trafficlight.getPhase("center")

    # Switch phase every 50 steps
    if step % 50 == 0:
        new_phase = (phase + 1) % 4
        traci.trafficlight.setPhase("center", new_phase)

    # Print queue (example)
    queue = traci.edge.getLastStepVehicleNumber("n2c")
    print(f"Step {step}, Queue: {queue}")

traci.close()