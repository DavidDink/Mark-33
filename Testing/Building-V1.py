from py4j.java_gateway import JavaGateway



gateway = JavaGateway()
test = gateway.entry_point
print (test.getStandardSeed())
test.setStandardSeed(1345)
#test.resetSession()
test.inputAction(0.0, 2.0)
array = test.getEnvironmentState()
print (list(array))
gateway.shutdown()

"""

This script is the environment part of our showcase.

"""



#Setup the Environment

class Environment:
    def __init__(self):
        gateway = JavaGateway()
        self.sim = gateway.entry_point
        self.sim.inputAction(0.0, 0.0)

    def shutdown(self):
        self.gateway.shutdown()

    def run(self, agent):
        self.sim.resetSession()

