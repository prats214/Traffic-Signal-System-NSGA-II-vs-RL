import traci

traci.start(["sumo-gui", "-c", "config.sumocfg"])

total_wait = 0
total_queue = 0

for step in range(1000):
    traci.simulationStep()

    total_wait += traci.edge.getWaitingTime("n2c")
    total_queue += traci.edge.getLastStepVehicleNumber("n2c")

    # Example: switch every 50 steps
    if step % 50 == 0:
        traci.trafficlight.setPhase("center", (step//50) % 4)
print("Avg Wait:", total_wait/1000)
print("Avg Queue:", total_queue/1000)

traci.close()