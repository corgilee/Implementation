'''

This implementation includes three classes:

Elevator: Represents an individual elevator in the building. It keeps track of its current floor, direction of movement, and pending requests.
ElevatorSystem: Manages multiple elevators in the building. It handles requests for elevators to specific floors and directs the nearest available elevator to handle the request.
Main: Contains the example usage of the ElevatorSystem class.
In this implementation, when a request for an elevator is made to go to a specific floor, 
the ElevatorSystem finds the nearest available elevator and directs it to handle the request by calling the request_floor and move methods of the Elevator class.
'''



class Elevator:
    def __init__(self, num_floors):
        self.num_floors = num_floors
        self.current_floor = 1
        self.direction = 'up'
        self.requests = set()

    def request_floor(self, floor):
        if floor == self.current_floor:
            print("Already on floor", floor)
        elif floor < 1 or floor > self.num_floors:
            print("Invalid floor number")
        else:
            self.requests.add(floor)

    def move(self):
        if self.requests:
            next_floor = min(self.requests) if self.direction == 'up' else max(self.requests)
            self.requests.remove(next_floor)
            print("Moving", self.direction, "to floor", next_floor)
            self.current_floor = next_floor
        else:
            print("No more requests, elevator is idle")

class ElevatorSystem:
    def __init__(self, num_elevators, num_floors):
        self.elevators = [Elevator(num_floors) for _ in range(num_elevators)]

    def request_elevator(self, floor, direction):
        elevator = self._find_elevator(floor)
        if elevator:
            elevator.request_floor(floor)
            elevator.direction = direction
            elevator.move()
        else:
            print("No available elevator")

    def _find_elevator(self, floor):
        for elevator in self.elevators:
            if elevator.current_floor == floor:
                return elevator
        return None

# Example usage:
if __name__ == "__main__":
    elevator_system = ElevatorSystem(num_elevators=2, num_floors=10)

    # Request elevator to go up from floor 3
    elevator_system.request_elevator(floor=3, direction='up')

    # Request elevator to go down from floor 8
    elevator_system.request_elevator(floor=8, direction='down')

    # Request elevator to go up from floor 6
    elevator_system.request_elevator(floor=6, direction='up')
