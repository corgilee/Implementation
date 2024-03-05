'''
Representation of Cards: Each card in the deck should be represented by its suit and rank. The deck should include all standard playing cards, which typically consist of four suits (hearts, diamonds, clubs, and spades) and thirteen ranks (2 through 10, Jack, Queen, King, and Ace).

Initialization: The deck should be initialized with a standard set of 52 playing cards, with one card for each combination of suit and rank.

Shuffling: The deck should provide a method to shuffle the cards randomly. Shuffling ensures that the cards are in a random order before dealing them to players.

Dealing Cards: The deck should provide a method to deal cards to players or to the table. Dealing involves distributing the cards from the top of the deck to the desired recipients.

Removing Cards: As cards are dealt or played, they should be removed from the deck to prevent them from being dealt again. Removing cards ensures that each card is dealt only once in a single game.

Resetting the Deck: After all the cards have been dealt or played, the deck should provide a method to reset itself to its original state, with all cards back in the deck and in a random order.

Handling Jokers (Optional): Some card games may include joker cards. If required, the deck should support the inclusion of joker cards and handle them appropriately during shuffling, dealing, and removing cards.

Efficient Operations: The deck should be designed to perform operations such as shuffling, dealing, and removing cards efficiently, especially for large numbers of cards.

Validation: The deck should include error checking and validation mechanisms to ensure that operations like dealing cards do not result in errors, such as dealing more cards than are available in the deck.

Encapsulation: The deck should be designed using object-oriented principles such as encapsulation, ensuring that the internal state of the deck is not directly accessible or modifiable from outside the deck object.
'''



# 以下是chatgpt 的 implementation + 我自己的改写

import numpy.random as nd 
class Card:
    #要制定 rank 和 suit
    def __init__(self,suit,rank):
        self.suit=suit
        self.rank=rank

class Deck:
    def __init__(self):
        self.reset()

    def shuffle(self):
        nd.shuffle(self.cards)

    def deal_card(self):
        if len(self.cards)==0:
            self.reset()
            print('*****The cards have been reset*****')

        top=self.cards.pop()
        #print(top)
        return top

        
    def reset(self):
        self.cards=[]
        suits=['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks=['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
        for suit in suits:
            for rank in ranks:
                self.cards.append(Card(suit, rank))

        # 要洗一下牌
        self.shuffle()

    def add_jokers(self):
        self.cards.append(Card('Red Joker','Red Joker'))
        self.cards.append(Card('Black Joker','Black Joker'))
        self.shuffle()


# test case:

deck=Deck()
deck.add_jokers()
#print(len(deck.cards))

for _ in range(55):
    card=deck.deal_card()
    print(f'{card.rank} of {card.suit}')
