from py4j.java_gateway import JavaGateway

class Environment:
    def __init__(self):
        self.gateway = JavaGateway()
        self.sim = self.gateway.entry_point

env = Environment()