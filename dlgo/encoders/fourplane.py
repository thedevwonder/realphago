from dlgo.encoders.base import Encoder
from dlgo.gotypes import Point, Player
import numpy as np

"""
Feature name            num of planes   Description
Stone colour            3               Player stone / opponent stone / empty
Ones                    1               A constant plane filled with 1
"""

FEATURE_OFFSETS = {
    "stone_color": 0,
    "ones": 3,
    "current_player_color": 4,
    "legal_moves": 5
}


def offset(feature):
    return FEATURE_OFFSETS[feature]


class FourplaneEncoder(Encoder):
    def __init__(self, board_size=(19, 19), use_player_plane=True, use_legal_moves=False):
        self.board_width, self.board_height = board_size
        self.use_player_plane = use_player_plane
        self.use_legal_moves = use_legal_moves
        self.num_planes = 4 + use_player_plane

    def name(self):
        return 'alphago'

    def encode(self, game_state):
        board_tensor = np.zeros((self.num_planes, self.board_height, self.board_width))
        
        # Set empty cells plane (default to empty)
        board_tensor[offset("stone_color") + 2] = 1
        
        # Iterate over occupied points only (much faster for sparse boards)
        next_player = game_state.next_player
        opponent = next_player.other
        stone_color_offset = offset("stone_color")
        
        for point, go_string in game_state.board._grid.items():
            if go_string is None:
                continue
            r = point.row - 1
            c = point.col - 1
            if go_string.color == next_player:
                board_tensor[stone_color_offset][r][c] = 1
                board_tensor[stone_color_offset + 2][r][c] = 0  # Not empty
            elif go_string.color == opponent:
                board_tensor[stone_color_offset + 1][r][c] = 1
                board_tensor[stone_color_offset + 2][r][c] = 0  # Not empty
        
        # Set ones plane once (moved outside loop)
        board_tensor[offset("ones")] = 1
        
        # Set player plane once (moved outside loop)
        if self.use_player_plane and next_player == Player.black:
            board_tensor[offset("current_player_color")] = 1
        

        return board_tensor

    def ones(self):
        return np.ones((1, self.board_height, self.board_width))

    def zeros(self):
        return np.zeros((1, self.board_height, self.board_width))

    def encode_point(self, point):
        return self.board_width * (point.row - 1) + (point.col - 1)

    def decode_point_index(self, index):
        row = index // self.board_width
        col = index % self.board_width
        return Point(row=row + 1, col=col + 1)

    def num_points(self):
        return self.board_width * self.board_height

    def shape(self):
        return self.num_planes, self.board_height, self.board_width


def create(board_size):
    return FourplaneEncoder(board_size)
