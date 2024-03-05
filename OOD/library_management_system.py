#Design a simple library management system

'''
book management
title, author, isbn, available

member management
name, email, 借出去的书

library management 
1. 录入书, 
2. 录入会员, 
3. 某会员借书 
4. 某会员还书

'''

class Book:
    def __init__(self, title, author,isbn):
        self.title=title
        self.author=author
        self.isbn=isbn
        self.available=True


class Member:
    def __init__(self, name, email):
        self.name=name
        self.email=email
        self.book_checkout=[]

class Library:
    def __init__(self):
        self.books=[]
        self.members=[]

    def add_book(self,book):
        #book=Book(title, author,isbn)
        self.books.append(book)
        print(f'{book.title} has been added to the system')

    def add_member(self,member):
        self.members.append(member)
        print(f'{member.name} has been added to the system')

    def checkout_book(self, book, member):
        #print(book.name)
        if book.available==True:
            book.available=False
            member.book_checkout.append(book)
            print(f'{member.name} has check out {book.title}')
            return True
        else:
            return False
    
    def return_book(self, book, member):
        if book in member.book_checkout:
            member.book_checkout.remove(book)
            book.available=True
            print(f'{member.name} has returned {book.title}')
            return True
        else:
            return False



# Example usage:
library=Library()

# Add books to the library
book1=Book("The Great Gatsby", "F. Scott Fitzgerald", "9780743273565")
book2=Book("To Kill a Mockingbird", "Harper Lee", "9780061120084")

library.add_book(book1)
library.add_book(book2)


# book1 = library.add_book("The Great Gatsby", "F. Scott Fitzgerald", "9780743273565")
# book2 = library.add_book("To Kill a Mockingbird", "Harper Lee", "9780061120084")


# Add members to the library

member1=Member("Alice", "alice@example.com")
member2=Member("Bob", "bob@example.com")

library.add_member(member1)
library.add_member(member2)


# Check out books
library.checkout_book(book1, member1)
library.checkout_book(book2, member2)

# Return books
library.return_book(book1, member1)
library.return_book(book2, member2)





        

        