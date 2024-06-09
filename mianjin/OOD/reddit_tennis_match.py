
'''
tennis, game, set, match

每一个game, set,match 的大致思路是一样的

class game
1. initialize, player, score_sequence=[0,15,30,40], 有一个player_points={p1:0,p2:0} 记录得分
2. point_by_player(self,player): 谁赢了就得1 分
3. score, 是print 分数的，把选手 points 和score_sequence 对应起来，score=min(score, 3)
4. is_winner(): 如果决出了winner (score>=4 and score gap>=2)，就返回winner，如果没有，就是None

class set
1. initialize, player, player_games={p1:0,p2:0} 这里记录的是每个player 当前set里面赢了多少个game 
,game_list=[] 记录每一个game的
2. start_a_new_game(self): initialize a new game and append to game list
3. point_by_player(self,player): 分情况讨论： 如果game_list里面没有任何game，或者game_list 里面的最后一个game是有赢家的，那么就是要新建一个game(start_a_new_game)
game_list[-1].point_by_player(player)
查看一下game_list[-1].is_winner() 是不是none，不是的话，要更新player_games


class match, 和class set 的思想是一样的，match 是五局三胜

'''


class Game:
    def __init__(self,p1,p2):
        self.player1=p1
        self.player2=p2
        self.player_points={self.player1:0,self.player2:0}
        self.score_list=[0,15,30,40]

    def point_by_player(self,player):
        #edge case
        if player not in self.player_points:
            raise ValueError("Invalid player")

        # add 1 more point to the player
        self.player_points[player]+=1

    def score(self):
        p1_points=self.player_points[self.player1]
        p2_points=self.player_points[self.player2]
        p1_score=self.score_list[min(p1_points,3)] #index cannot larger than 3
        p2_score=self.score_list[min(p2_points,3)] #index cannot larger than 3
        return f"current game score: {self.player1}-{self.player2}: {p1_score}-{p2_score}"

    def is_winner(self):
        p1_points=self.player_points[self.player1]
        p2_points=self.player_points[self.player2]
        if p1_points>=4 and p1_points-2>=p2_points:
            return self.player1
        elif p2_points>=4 and p2_points-2>=p1_points:
            return self.player2
        else:
            return


#-------------------------------------
# test example  Create a game
game = Game("player_1", "player_2")

# Simulate some points
game.point_by_player("player_1")
game.point_by_player("player_2")
game.point_by_player("player_1")
game.point_by_player("player_1")

#print(game.score())
#-------------------------------------

class Set:
    def __init__(self,p1,p2):
        self.player1=p1
        self.player2=p2
        self.game_list=[] # 包含了所有game， 每一个game是一个类
        self.player_games={self.player1:0,self.player2:0}

    def start_a_new_game(self):
        new_game=Game(self.player1,self.player2) #这里一定注意是self.player1, self.player2
        self.game_list.append(new_game)

    def point_by_player(self,player):
        # 先讨论情况，要不要start a new game
        if not self.game_list or self.game_list[-1].is_winner() is not None:
            self.start_a_new_game()
        #当前game，给player 加point
        self.game_list[-1].point_by_player(player)
        # 看一下当前game有没有winner，有的话，要加到player_games里面去
        winner=self.game_list[-1].is_winner()
        if winner:
            self.player_games[winner]+=1
    
    def score(self):
        p1_score=self.player_games[self.player1]
        p2_score=self.player_games[self.player2]

        return f"current set score: {self.player1}-{self.player2}: {p1_score}-{p2_score}"

    def is_winner(self):
        # 要win 6 games and 2 more games than the other
        p1_score=self.player_games[self.player1]
        p2_score=self.player_games[self.player2]

        if p1_score>=6 and p1_score-p2_score>=2:
            return self.player1
        elif p2_score>=6 and p2_score-p1_score>=2:
            return self.player2
        else:
            return None

#------- Text case: Create a set---------------
game_set = Set("player_1", "player_2")

while not game_set.is_winner():
    game_set.point_by_player("player_1")
    game_set.point_by_player("player_1")
    game_set.point_by_player("player_2")

# print(game_set.score())
# print(game_set.is_winner())

class Match:
    def __init__(self,p1,p2):
        self.player1=p1
        self.player2=p2
        self.player_sets={self.player1:0,self.player2:0}
        self.set_list=[] # 储存每一个set

    def start_a_new_set(self):
        new_set=Set(self.player1,self.player2)
        self.set_list.append(new_set) #开一个新的set

    def point_by_player(self,player):
        #先检查一下有没有set，或者当前set 有没有赢家
        if not self.set_list or self.set_list[-1].is_winner():
            self.start_a_new_set()
        #当前set，player 加分
        self.set_list[-1].point_by_player(player)
        # 查看有没有当前set 有没有winner
        winner=self.set_list[-1].is_winner()
        if winner:
            self.player_sets[player]+=1

    def score(self):
        p1_score=self.player_sets[self.player1]
        p2_score=self.player_sets[self.player2]

        return f"current match score: {self.player1}-{self.player2}: {p1_score}-{p2_score}"

    def is_winner(self):
        p1_score=self.player_sets[self.player1]
        p2_score=self.player_sets[self.player2]

        if p1_score>=3 and p1_score-p2_score>=2:
            return self.player1
        elif p2_score>=3 and p2_score-p1_score>=2:
            return self.player2
        else:
            return None
        
#------- Text case: Create a match---------------
match = Match("player_1", "player_2")

while not match.is_winner():
    match.point_by_player("player_1")
    match.point_by_player("player_2")
    match.point_by_player("player_1")
    match.point_by_player("player_2")
    match.point_by_player("player_1")

print(match.score())
print(match.set_list[0].score())
print(match.set_list[0].game_list[0].score())
print(match.is_winner())   












        