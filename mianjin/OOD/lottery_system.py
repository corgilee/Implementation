
import random

class LotteryTicket:
    def __init__(self, ticket_number, user):
        self.ticket_number = ticket_number
        self.user = user

    def __repr__(self):
        return f"Ticket(number={self.ticket_number}, user={self.user.name})"

class User:
    def __init__(self, name):
        self.name = name
        self.tickets = []

    def buy_ticket(self, lottery_system):
        ticket = lottery_system.issue_ticket(self)
        self.tickets.append(ticket)
        return ticket

    def __repr__(self):
        return self.name

class LotteryDraw:
    def __init__(self, lottery_system):
        self.lottery_system = lottery_system
        self.winning_number = None

    def draw_winning_number(self):
        if not self.lottery_system.tickets:
            print("No tickets sold. No draw can be performed.")
            return
        self.winning_number = random.choice(self.lottery_system.tickets).ticket_number
        print(f"The winning number is: {self.winning_number}")

    def check_winners(self):
        winners = [ticket for ticket in self.lottery_system.tickets if ticket.ticket_number == self.winning_number]
        if winners:
            for winner in winners:
                print(f"Congratulations {winner.user.name}, you have a winning ticket: {winner.ticket_number}!")
        else:
            print("No winning tickets this draw.")

class LotterySystem:
    def __init__(self):
        self.tickets = []
        self.ticket_counter = 1

    def issue_ticket(self, user):
        ticket = LotteryTicket(ticket_number=self.ticket_counter, user=user)
        self.tickets.append(ticket)
        self.ticket_counter += 1
        return ticket

    def draw(self):
        draw = LotteryDraw(self)
        draw.draw_winning_number()
        draw.check_winners()


# Create a lottery system
lottery_system = LotterySystem()


# Create users
alice = User("Alice")
bob = User("Bob")
charlie = User("Charlie")

# Users buy tickets
alice.buy_ticket(lottery_system)
alice.buy_ticket(lottery_system)
bob.buy_ticket(lottery_system)
charlie.buy_ticket(lottery_system)
charlie.buy_ticket(lottery_system)

# Perform a draw
lottery_system.draw()
