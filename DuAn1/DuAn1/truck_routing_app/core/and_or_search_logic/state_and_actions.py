class State:
    def __init__(self, location, is_broken_down=False):
        self.location = location
        self.is_broken_down = is_broken_down

    def clone(self):
        return State(self.location, self.is_broken_down)

    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        return (self.location == other.location and
                self.is_broken_down == other.is_broken_down)

    def __hash__(self):
        return hash((self.location, self.is_broken_down))

    def __str__(self):
        return f"State(Location: {self.location}, Broken: {self.is_broken_down})"

    def __repr__(self):
        return self.__str__()

class DriveAction:
    def __init__(self, destination):
        self.type = "DRIVE"
        self.destination = destination

    def __str__(self):
        return f"DriveAction(To: {self.destination})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, DriveAction):
            return NotImplemented
        return self.type == other.type and self.destination == other.destination

    def __hash__(self):
        return hash((self.type, self.destination))

class RepairAction:
    def __init__(self):
        self.type = "REPAIR"

    def __str__(self):
        return "RepairAction()"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, RepairAction):
            return NotImplemented
        return True # All RepairActions are the same

    def __hash__(self):
        return hash("REPAIR") 