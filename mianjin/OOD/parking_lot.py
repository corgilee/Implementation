
'''
The Vehicle class represents a vehicle with attributes like license plate and vehicle type.
The ParkingLot class represents a parking lot with a certain capacity and a list of spots where vehicles can be parked.
The park_vehicle method in the ParkingLot class is used to park a vehicle. It iterates over the spots in the parking lot 
and assigns the vehicle to the first available spot.
The remove_vehicle method in the ParkingLot class is used to remove a vehicle from a specific spot in the parking lot.
'''




class Vehicle:
    def __init__(self, license_plate, vehicle_type):
        self.license_plate = license_plate
        self.vehicle_type = vehicle_type

class ParkingLot:
    def __init__(self, capacity):
        self.capacity = capacity
        self.spots = [None] * capacity

    def park_vehicle(self, vehicle):
        for i in range(self.capacity):
            if self.spots[i] is None:
                self.spots[i] = vehicle
                return i  # Return the spot index where the vehicle is parked
        return -1  # Return -1 if parking lot is full

    def remove_vehicle(self, spot_index):
        if 0 <= spot_index < self.capacity:
            self.spots[spot_index] = None
            return True  # Vehicle removed successfully
        return False  # Invalid spot index

# Example usage:
if __name__ == "__main__":
    # Create a parking lot with capacity of 5
    parking_lot = ParkingLot(5)

    # Park vehicles
    vehicle1 = Vehicle("ABC123", "Car")
    spot1 = parking_lot.park_vehicle(vehicle1)
    print("Vehicle 1 parked at spot:", spot1)

    vehicle2 = Vehicle("XYZ789", "Motorcycle")
    spot2 = parking_lot.park_vehicle(vehicle2)
    print("Vehicle 2 parked at spot:", spot2)

    # Remove vehicle 1
    success = parking_lot.remove_vehicle(spot1)
    if success:
        print("Vehicle 1 removed successfully")
    else:
        print("Failed to remove vehicle 1")
