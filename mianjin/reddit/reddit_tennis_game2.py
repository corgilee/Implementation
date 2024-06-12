'''
两个人玩一个游戏，A 赢 则 A 积一分，完成以下函数：
1 输出两个人当前比分
2 积分函数，A 赢则积1分，B赢则B积分， 如果超过3比3 并且比分相同则重置比分为3：3.  如果已有获胜者 继续调这个函数则返回error。
3 winner 函数， 如果A 超过5分并且领先 B 2分 则A获胜输出A。 如果比赛没有获胜者，这个函数要返回 error

Part2
还是这个游戏 但是要玩5局。 谁先赢3局算赢。 更新并扩展‍‌‌‌‍‍‌‍‌‌‌‍‍‍‌‌‌‌你的函数。
Part3
improve
'''

#------ part 1 ----------
class Game:
    def __init__(self):
        self.score1=0
        self.score2=0
        self.winner=None

    def print_score(self):
        return f"current score between player_1 and player_2 is {self.score1}:{self.score2}"
    
    def point_by_player(self, player):
        if self.winner is not None:
            #raise ValueError("The game is already over")
            print("The game is already over")
            return 
        else:
            if player=="player_1":
                self.score1+=1
            elif player=="player_2":
                self.score2+=1
        if self.score1==self.score2 and self.score1>=3 and self.score2>=3:
            self.score1=3
            self.score2=3
        elif self.score1>=5 and self.score1-self.score2>=2:
            self.winner="player_1"
        elif self.score2>=5 and self.score2-self.score1>=2:
            self.winner="player_2"
            
    def who_is_winner(self):
        if self.winner is None:
            #raise ValueError("No winner for the game yet")
            return "No winner for the game yet"
        else:
            return f"The winner is {self.winner}"
game=Game()
game.point_by_player("player_1")
game.point_by_player("player_1")
game.point_by_player("player_1")
game.point_by_player("player_1")
game.point_by_player("player_2")
game.point_by_player("player_2")
game.point_by_player("player_1")
# print(game.print_score())
# print(game.who_is_winner())

#------ part 2 --------
# 新增一个set，init 里面存有 list of game
# 每次point_by_player 的时候，check一下last game 有没有winner, 如果已经有winner 就新开一个game
class Set:
    def __init__(self):
        self.game_list=[]
        self.winner=None
        self.game1=0
        self.game2=0

    def print_score(self):
        print(f"The current score between player1 and player2 is {self.game1} : {self.game2} ")
    
    def add_new_game(self):
        g=Game()
        self.game_list.append(g)
        

    def point_by_player(self,player):
        if self.winner is not None:
            print("The set has ended")
            return 
        if len(self.game_list)==0 or self.game_list[-1].winner is not None:
            self.add_new_game()
        

        self.game_list[-1].point_by_player(player)
        
        #要判断一下这个点之后game 有没有结束
        if self.game_list[-1].winner is not None:
            if self.game_list[-1].winner=="player_1":
                self.game1+=1
            else:
                self.game2+=1
        
        #要判断一下这个set 有没有结束
        if self.game1>=2 and self.game1-self.game2>=1:
            self.winner="player_1"
        elif self.game2>=2 and self.game2-self.game1>=1:
            self.winner="player_2"

    def who_is_winner(self):
        if self.winner is None:
            print("No winner yet")
        else:
            print(f"The winner is {self.winner}")


s=Set()
while s.winner is None:
    s.point_by_player("player_2")
    s.point_by_player("player_2")
    s.point_by_player("player_1")
  
s.print_score()
s.who_is_winner()     
        
        
        
        

                    
                    
            
    







